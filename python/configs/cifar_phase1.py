import argparse
import torch

def create_config_parser():
    parser = argparse.ArgumentParser(description='PyTorch Phase 1')

    parser.add_argument('--seed', type=int, default=999, help='')
    parser.add_argument("--random_rotation", type=int, default=100)
    parser.add_argument("--random_crop", type=int, default=50)
    parser.add_argument("--pretensor_transform", action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')
    parser.add_argument('--num_workers', type=int, default=10, help='')
    parser.add_argument('--save-model', action='store_true', default=False, help='')
    parser.add_argument('--best_threshold', type=float, default=0.5)
    parser.add_argument('--verbose', type=int, default=200, help='')

    parser.add_argument('--avoid_cls_reinit', action='store_true', default=True, help='')


    parser.add_argument('--dataset', type=str, default='celeba', help='')
    parser.add_argument('--batch-size', type=int, default=500, help='')

    parser.add_argument('--path', type=str, default='', help='')

    parser.add_argument('--data_root', type=str, default='/mnt/wjy-data/data/CELEBA')

    parser.add_argument('--clsmodel', type=str, default='ResNet18', help='')

    parser.add_argument('--epochs', type=int, default=6000, help='')
    parser.add_argument('--train-epoch', type=int, default=100, help='')
    parser.add_argument('--cls_test_epochs', type=int, default=5000, help='')
    parser.add_argument('--epochs_per_external_eval', type=int, default=1000, help='')
    parser.add_argument('--lr', type=float, default=0.5, help='')



    parser.add_argument('--mode', type=str, default='all2all', help='')
    parser.add_argument('--attack_model', type=str, default='autoencoder', help='')
    # only in effect if it's all to one
    parser.add_argument('--target_label', type=int, default=0)
    # eps
    parser.add_argument('--eps', type=float, default=0.05, help='')
    parser.add_argument('--alpha', type=float, default=0.5)
    # 投毒比例
    parser.add_argument('--attack_portion', type=float, default=1.0, help='')
    parser.add_argument('--lr-atk', type=float, default=0.5, help='')


    # 相似性参数 ---------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--mul_similarity', type=float, default=0, help='')
    parser.add_argument('--alpha_similarity', type=float, default=0.9, help='')

    # test -------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--test_epochs', type=int, default=10000, help='')
    parser.add_argument('--test_eps', type=float, default=0.5, help='')
    parser.add_argument('--test_alpha', type=float, default=0.5)
    parser.add_argument('--test_attack_portion', type=float, default=10.0)
    parser.add_argument('--test_lr', type=float, default=0.5)
    parser.add_argument('--test_n_size', type=int, default=1000)
    parser.add_argument('--test_img_row', type=int, default=1000)
    parser.add_argument('--test_optimizer', type=str, default='sgd', help='')

    parser.add_argument('--test_use_train_best', default=True, action='store_true')
    parser.add_argument('--test_use_train_last', default=True, action='store_true')

    parser.add_argument('--use_data_parallel', default=True, action='store_true')

    parser.add_argument('--schedulerC_lambda', type=float, default=0.5)
    parser.add_argument('--schedulerC_milestones', type=str, default='')


    return parser