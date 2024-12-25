import argparse
import sys
import os

sys.path.append('.')
from utils.dataset import ProteinIntDataset, saveDatasetOnDisk
from utils.tool import getConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--saveroot", default="", type=str)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()

    cfg = getConfig(args.config)

    if 'cuda' in cfg['basic']['embedder_device']:
        num_workers = 1
    else:
        num_workers = args.workers

    dataset = ProteinIntDataset(
        path=cfg['dataset'][args.split]['path'], 
        data_root=cfg['dataset'][args.split]['data_root'],
        embedding=cfg['basic']['embedding'],
        distance_threshold=cfg['basic']['distance_threshold'],
        use_cuda=cfg['basic']['embedder_device']
    )

    if args.saveroot == '' or args.saveroot is None:
        args.saveroot = os.path.join('datasets', args.split)

    saveDatasetOnDisk(dataset, args.saveroot, args.workers)
