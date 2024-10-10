import argparse

import utils.ctrl as ctrl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="train", help="Target to do")
    parser.add_argument("--config", default="./config/tsubame.yaml", help="The config file to use")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print("Using confg file", args.config)

    if args.task == 'train_interface':
        print('Train model')
        ctrl.train_interface(args.config, resume=args.resume)
    elif args.task == 'test_interface':
        print('Evaluate model')
        ctrl.eval_interface(args.config)
    elif args.task == 'inference_interface':
        print('Inference model')
        ctrl.inference_interface(args.config)
    elif args.task == 'example_interface':
        ctrl.example_interface(args.config)
    elif args.task == 'train_interaction':
        ctrl.train_interaction(args.config, resume=args.resume)
    elif args.task == 'eval_interaction':
        ctrl.eval_interaction(args.config)
