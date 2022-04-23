import torch
import os
import argparse


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in args.gpus)




def parse_args():
    parser = argparse.ArgumentParser(description='Train CLRNet')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--view', action='store_true',
        help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int,
                        default=0, help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    main()