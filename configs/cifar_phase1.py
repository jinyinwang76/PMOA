import argparse
import torch
import os
os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
def create_config_parser():
    parser = argparse.ArgumentParser(description='PyTorch Phase 1')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_root', type=str, default='data/CIFAR/')
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--pretensor_transform", action='store_true', default=False)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='training device')
    parser.add_argument('--num-workers', type=int, default=2, help='dataloader workers')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-atk', type=float, default=0.0001, help='learning rate for attack model')
    parser.add_argument('--seed', type=int, default=999, help='random seed (default: 999)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--train-epoch', type=int, default=1, help='training epochs for victim model')
    # only in effect if it's all to one
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.3, help='epsilon for data poisoning')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--clsmodel', type=str, default='vgg11')
    parser.add_argument('--attack_model', type=str, default='autoencoder')
    parser.add_argument('--attack_portion', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='all2one')
    parser.add_argument('--epochs_per_external_eval', type=int, default=50)
    parser.add_argument('--cls_test_epochs', type=int, default=20)
    parser.add_argument('--path', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--best_threshold', type=float, default=0.1)
    parser.add_argument('--verbose', type=int, default=1, help='verbosity')
    parser.add_argument('--avoid_cls_reinit', action='store_true', default=False, help='whether test the poisoned model from scratch')

    parser.add_argument('--test_eps', default=None, type=float)
    parser.add_argument('--test_alpha', default=None, type=float)

    return parser