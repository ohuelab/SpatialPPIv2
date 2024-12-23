import argparse
import numpy as np

from utils.ctrl import train, eval
from utils.tool import getConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="train", help="Train or eval the model")
    parser.add_argument("--config", default="config/default.yaml", help="The config file to use")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint / Evaluate target checkpoint")
    parser.add_argument("--output", type=str, help="Save output to file, end with .npy")
    args = parser.parse_args()

    print("Using confg file", args.config)

    cfg = getConfig(args.config)

    if args.task == 'train':
        train(cfg, args.checkpoint)
    elif args.task == 'eval':
        pred, label = eval(cfg, args.checkpoint)
        if args.output:
            np.save(args.output, pred.numpy())

