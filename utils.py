import os

def get_checkpoints(checkpoint_path):
    checkpoint_files = [f for dp, dn, filenames in os.walk(checkpoint_path) for f in filenames if os.path.splitext(f)[-1] == '.pt']
    checkpoints = {}
    for checkpoint_file in checkpoint_files:
        checkpoints[int(checkpoint_file[11:-3])] = checkpoint_file
    return checkpoints

def update_checkpoints(checkpoint_path, checkpoints, keep_checkpoints):
    checkpoint_steps = list(checkpoints.keys())
    checkpoint_steps.sort()
    checkpoint_steps = checkpoint_steps[:- keep_checkpoints]
    for checkpoint_step in checkpoint_steps:
        os.remove(os.path.join(checkpoint_path, checkpoints[checkpoint_step]))
        checkpoints.pop(checkpoint_step)