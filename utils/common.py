import sys
import os
import os.path as osp
import math
import time
import requests
import zipfile, tarfile, gzip
import torch
from glob import glob
from tqdm import tqdm


def download_file(url, filepath):
    """Downloads a file from the given URL."""
    print("Downloading %s..." % url)
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(filepath, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='MB'):
            wrote = wrote + len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        print("Downloading failed")
        sys.exit(1)

def extract_gzfile(filepath, dstdir='data'):
    os.makedirs(dstdir, exist_ok=True)
    filename = osp.basename(filepath)
    print('Extracting {}...'.format(filename))
    gz = gzip.GzipFile(filepath, 'r')
    filename = filename.replace('.gz', '')
    open(osp.join(dstdir, filename), 'w+').write(gz.read())
    gz.close()

def extract_zipfile(filepath, dstdir='data'):
    os.makedirs(dstdir, exist_ok=True)
    filename = osp.basename(filepath)
    print('Extracting {}...'.format(filename))
    zip = zipfile.ZipFile(filepath, 'r')
    zip.extractall(dstdir)
    zip.close()

def extract_tarfile(filepath, dstdir='data'):
    os.makedirs(dstdir, exist_ok=True)
    filename = osp.basename(filepath)
    print('Extracting {}...'.format(filename))
    tar = tarfile.TarFile(filepath, 'r')
    tar.extractall(dstdir)
    tar.close()

def get_last_checkpoint(dstdir):
    """Returns the last checkpoint file name in the given dstdir path."""
    checkpoints = glob(osp.join(dstdir, '*.pth'))
    checkpoints.sort()
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]

def save_checkpoint(logdir, epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state_dict = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'loss': loss,
        'model': model,
        'optimizer': optimizer
    }
    checkpoint_file_name = 'final_checkpoint.pth'
    torch.save(state_dict, osp.join(logdir, checkpoint_file_name))
    print(f"Saved the checkpoint (epoch={epoch:04d}) to '{checkpoint_file_name}'")
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state_dict, osp.join(logdir, 'best_checkpoint.pth'))
        print(f"Saved the checkpoint (epoch={epoch:04d}) to 'best_checkpoint.pth'")

def load_checkpoint(logdir, checkpoint_file_name=None):
    """Loads the checkpoint into the given model and optimizer."""
    checkpoint_file_name = checkpoint_file_name \
        if checkpoint_file_name is None else 'final_checkpoint.pth'
    checkpoint = torch.load(osp.join(logdir, checkpoint_file_name))
    epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    loss = checkpoint['loss']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    print(f"Loaded the checkpoint (epoch={epoch:04d}) from '{checkpoint_file_name}'")
    return epoch, epochs_since_improvement, model, optimizer, loss

