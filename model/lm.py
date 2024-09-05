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


class CodebookBlock(nn.Module):
    def __init__(self, layer_id: int, configs):
        super().__init__()
        self.n_heads = configs['model']['codebook_layers']['n_heads']
        self.dim = configs['model']['codebook_layers']['transformer_dim']
        self.head_dim = configs['model']['codebook_layers']['transformer_dim'] // configs['model']['codebook_layers']['n_heads']
        self.attention = Attention(configs)
        self.feed_forward = FeedForward(
            dim = configs['model']['codebook_layers']['transformer_dim'],
            hidden_dim = 4 * configs['model']['codebook_layers']['transformer_dim'],
            multiple_of = configs['model']['codebook_layers']['multiple_of'],
            ffn_dim_multiplier = None,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            dim = configs['model']['codebook_layers']['transformer_dim'], 
            eps = configs['model']['codebook_layers']['norm_eps']
        )
        self.ffn_norm = RMSNorm(
            dim = configs['model']['codebook_layers']['transformer_dim'], 
            eps = configs['model']['codebook_layers']['norm_eps']
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
        self.codebook_ids = configs['model']['codebook_ids']
        self.max_seq_len = configs['max_seq_len']
        self.frame_ratio = configs['model']['frame_ratio']
        self.codec_prompt_len = int(configs['model']['prompt_len'] / (configs['model']['frame_ratio'] + 1) * configs['model']['frame_ratio'])
        self.top_k = configs['training']['top_k']
        self.temperature = configs['training']['temperature']
        self.inference_sampler = torch.nn.Softmax(dim = 2)

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

        self.codebook_layers = torch.nn.ModuleList()
        for layer_id in range(configs['model']['codebook_num']):
            self.codebook_layers.append(TransformerBlock(layer_id, configs))

        self.norm = RMSNorm(
            dim = configs['model']['transformer_dim'], 
            eps = configs['model']['norm_eps']
        )
        
        self.output = torch.nn.ModuleList()
        for codebook_id in range(configs['model']['codebook_num']):
            self.output.append(torch.nn.Linear(
                in_features = configs['model']['transformer_dim'], 
                out_features = self.codebook_dim, 
                bias = False
            )
        )

        self.freqs_cis = precompute_freqs_cis(
            dim = configs['model']['transformer_dim'] // configs['model']['n_heads'],
            end = self.max_seq_len * 2,
            theta = configs['model']['rope_theta'],
        )

        self.codebook_freqs_cis = precompute_freqs_cis(
            dim = configs['model']['codebook_layers']['transformer_dim'] // configs['model']['codebook_layers']['n_heads'],
            end = self.max_seq_len * 2,
            theta = configs['model']['codebook_layers']['rope_theta'],
        )

        self.dropout_ratio = configs['training']['input_dropout']
        self.input_dropout = torch.nn.Dropout(p = self.dropout_ratio)

    def forward(self, codecs, asr_embs):
        # codecs: [batch_size, codec_seq_len]
        # asr_embs: [batch_size, asr_seq_len, emb_dim]
        # codec_seq_len = asr_seq_len * frame_ratio + 1

        batch_size = codecs.shape[0]
        codec_emb_codebooks = self.embeddings(codecs) # [batch_size, codec_seq_len, codebook_num, transformer_dim]
        
        # Apply frame-level dropout beyond the prompt
        frame_dropout_ref = torch.full([codec_emb_codebooks.shape[0], codec_emb_codebooks.shape[1] - self.codec_prompt_len, codec_emb_codebooks.shape[2]], 1.).to(codecs.device)
        frame_dropout_ref = self.input_dropout(frame_dropout_ref)
        # if not self.training:
        #     frame_dropout_ref = torch.div(frame_dropout_ref, 1 - self.dropout_ratio)
        frame_dropout_ref = torch.cat([torch.full([codec_emb_codebooks.shape[0], self.codec_prompt_len, codec_emb_codebooks.shape[2]], 1.).to(codecs.device), frame_dropout_ref], dim = 1)
        codec_emb_codebooks = codec_emb_codebooks * frame_dropout_ref.unsqueeze(-1)
        
        codec_embs = torch.sum(codec_emb_codebooks.clone(), dim = 2) # [batch_size, codec_seq_len, transformer_dim]
        # codec_embs = torch.sum(self.embeddings(codecs[:, :, self.codebook_ids]), dim = 2) # [batch_size, codec_seq_len, transformer_dim]

        filler_embs = torch.zeros([codec_emb_codebooks.shape[0], int((codec_emb_codebooks.shape[1] - 1) / self.frame_ratio), codec_emb_codebooks.shape[2], codec_emb_codebooks.shape[3]]).to(codecs.device)
        codec_emb_codebooks_final = codec_emb_codebooks[:, -1, :, :]
        codec_emb_codebooks = codec_emb_codebooks[:, :-1, :].view((codec_emb_codebooks.shape[0], int((codec_emb_codebooks.shape[1] - 1) / self.frame_ratio), self.frame_ratio, codec_emb_codebooks.shape[2], codec_emb_codebooks.shape[3]))
        codebook_embs = torch.cat([filler_embs.unsqueeze(2), codec_emb_codebooks], dim = 2)
        codebook_embs = codebook_embs.view((codebook_embs.shape[0], codebook_embs.shape[1] * codebook_embs.shape[2], codebook_embs.shape[3], codebook_embs.shape[4]))
        codebook_embs = torch.cat([codebook_embs, codec_emb_codebooks_final.unsqueeze(1)], dim = 1)
        codebook_embs = codebook_embs[:, :self.max_seq_len, :, :]

        codec_embs_final = codec_embs[:, -1, :] # extract the added last frame, so that seq_len is a multiple of frame_ratio
        codec_embs = codec_embs[:, :-1, :].view((codec_embs.shape[0], int((codec_embs.shape[1] - 1) / self.frame_ratio), self.frame_ratio, codec_embs.shape[2])) # reshape so that there is a dimension of length = frame_ratio
        # [batch_size, (codec_seq_len - 1)/frame_ratio, frame_ratio, transformer_dim]

        asr_embs = self.projection(asr_embs) # [batch_size, asr_seq_len, transformer_dim]
        embs = torch.cat([asr_embs.unsqueeze(2), codec_embs], dim = 2) # [batch_size, asr_seq_len, frame_ratio + 1, transformer_dim]
        del asr_embs
        del codec_embs
        embs = embs.view((embs.shape[0], embs.shape[1] * embs.shape[2], embs.shape[3])) # [batch_size, asr_seq_len * (frame_ratio + 1), transformer_dim]
        embs = torch.cat([embs, codec_embs_final.unsqueeze(1)], dim = 1) # [batch_size, asr_seq_len * (frame_ratio + 1) + 1, transformer_dim]
        embs = embs[:, :self.max_seq_len, :] # [batch_size, seq_len, transformer_dim]
        # seq_len = min(max_seq_len, asr_seq_len * (frame_ratio + 1) + 1)
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
        outputs = []
        next_token_codebooks = []
        for codebook_index, codebook_layer in enumerate(self.codebook_layers):
            embs = codebook_layer(embs, 0, freqs_cis, mask)
            out_embs = self.norm(embs.clone())
            output = self.output[codebook_index](out_embs).float() # [batch_size, seq_len, codebook_dim]
            outputs.append(output)
            if self.training:
                embs[:, :-1, :] += codebook_embs[:, 1:, codebook_index, :]
            else:
                next_token_candidates = torch.topk(output, k = self.top_k, dim = 2)
                next_token_probs = self.inference_sampler(torch.div(next_token_candidates.values, self.temperature))
                next_token_probs = next_token_probs.reshape(next_token_probs.shape[0] * next_token_probs.shape[1], next_token_probs.shape[2])
                next_token_candidate_indices = torch.multinomial(next_token_probs, 1).squeeze(1)
                next_token_candidates = next_token_candidates.indices
                next_token_candidates = next_token_candidates.reshape(next_token_candidates.shape[0] * next_token_candidates.shape[1], next_token_candidates.shape[2])
                next_tokens = torch.diagonal(torch.index_select(next_token_candidates, dim = 1, index = next_token_candidate_indices), dim1 = 0, dim2 = 1).view(output.shape[0], -1)
                
                next_tokens += codebook_index * self.codebook_dim
                
                next_token_embs = self.embeddings(next_tokens)
                embs += next_token_embs
                next_token_codebooks.append(next_tokens)

                del next_token_candidates
                del next_token_probs
                del next_token_candidate_indices
                del next_token_embs

        output = torch.stack(outputs, dim = -1) # [batch_size, seq_len, codebook_dim, codebook_num]
        if self.training:
            next_token_codebooks = None
        else:
            next_token_codebooks = torch.stack(next_token_codebooks, dim = 2)
        del embs
        del codec_emb_codebooks
        del codebook_embs
        del mask
        del outputs
        return output, next_token_codebooks