import os

def get_checkpoints(checkpoint_path):
    checkpoint_files = [f for dp, dn, filenames in os.walk(checkpoint_path) for f in filenames if os.path.splitext(f)[-1] == '.pt']
    checkpoints = {}
    for checkpoint_file in checkpoint_files:
        checkpoints[int(checkpoint_file[11:-3])] = checkpoint_file
    return checkpoints