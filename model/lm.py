# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    max_batch_size: int = 32


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device = freqs.device, dtype = torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_kv_heads = configs['model']['n_heads']
        model_parallel_size = 1 # todo: verify
        self.n_local_heads = configs['model']['n_heads'] // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = configs['model']['transformer_dim'] // configs['model']['n_heads']
        self.batch_size = configs['batch_size']
        self.max_seq_len = configs['max_seq_len']

        self.wq = torch.nn.Linear(
            in_features = configs['model']['transformer_dim'],
            out_features = configs['model']['n_heads'] * self.head_dim,
            bias = False,
        )
        self.wk = torch.nn.Linear(
            in_features = configs['model']['transformer_dim'],
            out_features = self.n_kv_heads * self.head_dim,
            bias = False,
        )
        self.wv = torch.nn.Linear(
            in_features = configs['model']['transformer_dim'],
            out_features = self.n_kv_heads * self.head_dim,
            bias = False,
        )
        self.wo = torch.nn.Linear(
            in_features = configs['model']['n_heads'] * self.head_dim,
            out_features = configs['model']['transformer_dim'],
            bias = False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis = freqs_cis)

        keys = xk[:bsz, : start_pos + seqlen]
        values = xv[:bsz, : start_pos + seqlen]

        # # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(
        #     keys, self.n_rep
        # )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(
        #     values, self.n_rep
        # )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(
            dim, hidden_dim, bias = False,
        )
        self.w2 = torch.nn.Linear(
            hidden_dim, dim, bias = False,
        )
        self.w3 = torch.nn.Linear(
            dim, hidden_dim, bias = False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, configs):
        super().__init__()
        self.n_heads = configs['model']['n_heads']
        self.dim = configs['model']['transformer_dim']
        self.head_dim = configs['model']['transformer_dim'] // configs['model']['n_heads']
        self.attention = Attention(configs)
        self.feed_forward = FeedForward(
            dim = configs['model']['transformer_dim'],
            hidden_dim = 4 * configs['model']['transformer_dim'],
            multiple_of = configs['model']['multiple_of'],
            ffn_dim_multiplier = None,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            dim = configs['model']['transformer_dim'], 
            eps = configs['model']['norm_eps']
        )
        self.ffn_norm = RMSNorm(
            dim = configs['model']['transformer_dim'], 
            eps = configs['model']['norm_eps']
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class StreamVoice(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.codebook_num = configs['model']['codebook_num']
        self.codebook_dim = configs['model']['codebook_dim']
        self.max_seq_len = configs['max_seq_len']
        self.frame_ratio = configs['model']['frame_ratio']

        self.embeddings = torch.nn.Embedding(
            num_embeddings = self.codebook_num * self.codebook_dim + 1, 
            embedding_dim = configs['model']['transformer_dim'],
            padding_idx = self.codebook_num * self.codebook_dim
        )

        self.projection = torch.nn.Linear(
            in_features = configs['model']['emb_dim'], 
            out_features = configs['model']['transformer_dim']
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(configs['model']['n_layers']):
            self.layers.append(TransformerBlock(layer_id, configs))

        self.norm = RMSNorm(
            dim = configs['model']['transformer_dim'], 
            eps = configs['model']['norm_eps']
        )
        
        self.output = torch.nn.Linear(
            in_features = configs['model']['transformer_dim'], 
            out_features = self.codebook_dim * self.codebook_num, 
            bias = False
        )

        self.freqs_cis = precompute_freqs_cis(
            dim = configs['model']['transformer_dim'] // configs['model']['n_heads'],
            end = self.max_seq_len * 2,
            theta = configs['model']['rope_theta'],
        )

        self.input_dropout = torch.nn.Dropout(p = configs['training']['input_dropout'])

    def forward(self, codecs, asr_embs):
        # codecs: [batch_size, codec_seq_len]
        # asr_embs: [batch_size, asr_seq_len, emb_dim]
        # codec_seq_len = asr_seq_len * frame_ratio + 1

        batch_size = codecs.shape[0]
        codec_embs = torch.sum(self.embeddings(codecs), dim = 2) # [batch_size, codec_seq_len, transformer_dim]
        
        # Apply frame-level dropout
        frame_dropout_ref = torch.full([codec_embs.shape[0], codec_embs.shape[1]], 1.).to(codecs.device)
        frame_dropout_ref = self.input_dropout(frame_dropout_ref)
        codec_embs = codec_embs * frame_dropout_ref.unsqueeze(-1)

        codec_embs_final = codec_embs[:, -1, :] # extract the added last frame, so that seq_len is a multiple of frame_ratio
        codec_embs = codec_embs[:, :-1, :].view((codec_embs.shape[0], int((codec_embs.shape[1] - 1) / self.frame_ratio), self.frame_ratio, codec_embs.shape[2])) # reshape so that there is a dimension of length = frame_ratio
        # [batch_size, (codec_seq_len - 1)/frame_ratio, frame_ratio, transformer_dim]

        asr_embs = self.projection(asr_embs) # [batch_size, asr_seq_len, transformer_dim]
        embs = torch.cat([asr_embs.unsqueeze(2), codec_embs], dim = 2) # [batch_size, asr_seq_len, frame_ratio + 1, transformer_dim]
        embs = embs.view((embs.shape[0], embs.shape[1] * embs.shape[2], embs.shape[3])) # [batch_size, asr_seq_len * (frame_ratio + 1), transformer_dim]
        embs = torch.cat([embs, codec_embs_final.unsqueeze(1)], dim = 1) # [batch_size, asr_seq_len * (frame_ratio + 1) + 1, transformer_dim]
        embs = embs[:, :self.max_seq_len, :] # [batch_size, max_seq_len, transformer_dim]
        seq_len = embs.shape[1]
        self.freqs_cis = self.freqs_cis.to(embs.device)
        freqs_cis = self.freqs_cis[:min(seq_len, self.max_seq_len)]

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device = embs.device)

            mask = torch.triu(mask, diagonal = 1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seq_len, 0), device = embs.device), mask]
            ).type_as(embs)

        for layer in self.layers:
            embs = layer(embs, 0, freqs_cis, mask)
        embs = self.norm(embs)
        output = self.output(embs).float() # [batch_size, seq_len, codebook_num * codebook_dim]
        output = output.view((output.shape[0], output.shape[1], self.codebook_dim, self.codebook_num)) # [batch_size, seq_len, codebook_dim, codebook_num]
        return output