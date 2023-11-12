from torch import nn
import torch
from sys import path
import pickle
import os

import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np

import seaborn as sns
from tqdm import tqdm
from termcolor import colored

import pathlib
from argparse import Namespace
import argparse

import torchvision
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as opt
import torch.nn.functional as F
import time

from configs.cifar_phase1 import create_config_parser
from utils.backdoor import get_target_transform
from utils.dataloader import get_dataloader, PostTensorTransform
from utils.min_norm_solvers import MGDASolver


def get_model(args, model_only=False):
    atkmodel = None
    netC = None
    optimizerC = None
    schedulerC = None

    # 触发器
    if args.dataset == 'cifar10':
        if args.attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(args.device)
        elif args.attack_model == 'autoencoder':
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder().to(args.device)
        else:
            raise Exception(f'Invalid generator model: {args.dataset}_{args.attack_model}')

    elif args.dataset == 'mnist':
        if args.attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(args.device)


        elif args.attack_model == 'autoencoder':
            from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
            atkmodel = Autoencoder().to(args.device)

        else:
            raise Exception(f'Invalid generator model: {args.dataset}_{args.attack_model}')

    elif args.dataset == 'tiny-imagenet' or args.dataset == 'tiny-imagenet32' or args.dataset == 'gtsrb':
        if args.attack_model == 'autoencoder':
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder().to(args.device)


        elif args.attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(args.device)

        else:
            raise Exception(f'Invalid generator model: {args.dataset}_{args.attack_model}')

    else:
        raise Exception(f'Invalid atk model {args.dataset}')


    if args.dataset == "cifar10" or args.dataset == "gtsrb":
        if args.clsmodel is None or args.clsmodel == 'PreActResNet18':
            from classifier_models import PreActResNet18
            netC = PreActResNet18(num_classes=args.num_classes).to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.dataset}_{args.clsmodel}")

    elif args.dataset == 'tiny-imagenet':
        if args.clsmodel is None or args.clsmodel == 'ResNet18TinyImagenet':
            from classifier_models import ResNet18TinyImagenet
            netC = ResNet18TinyImagenet().to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes, feature_dim=2048).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.dataset}_{args.clsmodel}")

    elif args.dataset == 'tiny-imagenet32':

        if args.clsmodel is None or args.clsmodel == 'ResNet18TinyImagenet':
            from classifier_models import ResNet18TinyImagenet
            netC = ResNet18TinyImagenet().to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes, feature_dim=512).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.dataset}_{args.clsmodel}")

    elif args.dataset == "mnist":
        if args.clsmodel is None or args.clsmodel == 'mnist_cnn':
            from networks.models import NetC_MNIST
            netC = NetC_MNIST().to(args.device)
        else:
            raise Exception(f"Invalid model version: {args.dataset}_{args.clsmodel}")

    elif args.dataset == 'celelba':
        if args.clsmodel is None or args.clsmodel == 'ResNet18':
            from classifier_models import ResNet18
            netC = ResNet18().to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes, feature_dim=2048).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.clsmodel}")

    else:
        raise Exception(f'Invalid model version: {args.dataset}_{args.clsmodel}')



    if model_only:
        return atkmodel, netC

    else:
        # # Optimizer
        # optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, momentum=0.9, weight_decay=5e-4)
        #
        # # Scheduler
        # schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)

        if args.test_optimizer == 'adam':
            print('使用 adam 优化器')
            optimizerC = torch.optim.Adam(netC.parameters(), args.test_lr, weight_decay=5e-4)
            # Scheduler
            schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)

        elif args.test_optimizer == 'sgd':
            print('使用 sgd 优化器')
            # optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, weight_decay=5e-4)
            optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, weight_decay=5e-4, momentum=0.9)
            # Scheduler
            schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)

        return atkmodel, netC, optimizerC, schedulerC


def final_test(args, test_model_path, atkmodel, netC, target_transform, train_loader, test_loader, trainepoch, writer, alpha=0.5, optimizerC=None, schedulerC=None,
               log_prefix='Internal', epochs_per_test=1, data_transforms=None, start_epoch=1, clip_image=None):
    atkmodel.eval()

    if optimizerC is None:
        print('No optimizer, creating default SGD...')
        optimizerC = opt.SGD(netC.parameters(), lr=args.test_lr)

    if schedulerC is None:
        print('No scheduler, creating default 100,200,300,400...')
        schedulerC = opt.lr_scheduler.MultiStepLR(optimizerC, [100, 200, 300, 400], args.test_lr)

    for cepoch in range(start_epoch, trainepoch + 1):

        netC.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=240)

        # 训练 --------------------------------------------------------------------------------------------------
        for batch_idx, (data, target) in pbar:
            bs = data.size(0)
            data, target = data.to(args.device), target.to(args.device)

            if data_transforms is not None:
                data = data_transforms(data)

            optimizerC.zero_grad()

            output = netC(data)
            loss_clean = F.cross_entropy(output, target)

            if alpha < 1:
                with torch.no_grad():

                    noise = atkmodel(data) * args.test_eps

                    if clip_image is None:
                        atkdata = torch.clamp(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)

                    atktarget = target_transform(target)

                    if args.test_attack_portion < 1.0:
                        atkdata = atkdata[:int(args.test_attack_portion * bs)]
                        atktarget = atktarget[:int(args.test_attack_portion * bs)]

                # tensor.detach() : requirse_grad 为false.得到的这个 tensor 永远不需要计算器梯度，不具有grad。
                atkoutput = netC(atkdata.detach())
                loss_poison = F.cross_entropy(atkoutput, atktarget.detach())
            else:
                loss_poison = torch.tensor(0.0).to(args.device)

            # 多目标优化 -------------------------------------------------------------------------------------------
            loss_values = {}
            loss_values['clean'] = loss_clean
            loss_values['poison'] = loss_poison

            grads = {}
            grads['clean'] = list(torch.autograd.grad(loss_values['clean'].mean(),
                                                      [x for x in netC.parameters() if x.requires_grad],
                                                      retain_graph=True))
            grads['poison'] = list(torch.autograd.grad(loss_values['poison'].mean(),
                                                       [x for x in netC.parameters() if x.requires_grad],
                                                       retain_graph=True))

            scale = MGDASolver.get_scales(grads, loss_values, 'none', ['clean', 'poison'])

            loss = scale['poison'] * loss_values['clean'] + scale['clean'] * loss_values['poison']
            # --------------------------------------------------------------------------------------------

            # loss = alpha * loss_clean + (1 - alpha) * loss_poison

            loss.backward()
            optimizerC.step()

            # if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
            #     pbar.set_description('Train-{} Loss: Clean {:.5f}  Poison {:.5f}  Total {:.5f} LR={:.6f}'.format(
            #                                                             cepoch,
            #                                                             loss_clean.item(),
            #                                                             loss_poison.item(),
            #                                                             loss.item(), schedulerC.get_last_lr()[0]
            #                                                         ))

            if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
                pbar.set_description('Train-{} Loss: Clean {:.5f}  Poison {:.5f}  Scales:{}  Total {:.5f} LR={:.6f}'.format(
                                                                        cepoch,
                                                                        loss_clean.item(),
                                                                        loss_poison.item(),
                                                                        scale,
                                                                        loss.item(), schedulerC.get_last_lr()[0]
                                                                    ))

        schedulerC.step()

        # 测试 --------------------------------------------------------------------------------------------------
        clean_accs, poison_accs = [], []
        best_clean_acc, best_poison_acc = 0, 0
        if cepoch % epochs_per_test == 0 or cepoch == trainepoch - 1:
            correct = 0
            correct_transform = 0
            test_loss = 0
            test_transform_loss = 0

            with torch.no_grad():
                for data, target in tqdm(test_loader, desc=f'Evaluation {cepoch}'):
                    # clean
                    data, target = data.to(args.device), target.to(args.device)
                    output = netC(data)

                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    # poison
                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = torch.clamp(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                    atkoutput = netC(atkdata)
                    test_transform_loss += F.cross_entropy(atkoutput, target_transform(target), reduction='sum').item()
                    atkpred = atkoutput.max(1, keepdim=True)[1]
                    correct_transform += atkpred.eq(target_transform(target).view_as(atkpred)).sum().item()

                    if cepoch == 1:
                        noise_original = atkmodel(data)
                        noise_process = noise

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)

            clean_accs.append(correct)
            poison_accs.append(correct_transform)

            print('\n{}-Test [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.4f} (best {:.4f}) poison {:.4f} (best {:.4f})'.format(
                log_prefix, cepoch,
                test_loss, test_transform_loss,
                correct, best_clean_acc, correct_transform, best_poison_acc
            ))

            if writer is not None:
                writer.add_scalar(f'{log_prefix}-Loss(clean)--{trainepoch}', test_loss, global_step=cepoch)
                writer.add_scalar(f'{log_prefix}-Loss(poison)--{trainepoch}', test_transform_loss, global_step=cepoch)
                writer.add_scalar(f'{log_prefix}-acc(clean)--{trainepoch}', correct, global_step=cepoch)
                writer.add_scalar(f'{log_prefix}-acc(poison)--{trainepoch})', correct_transform, global_step=cepoch)

            if correct > best_clean_acc or (correct > best_clean_acc - 0.02 and correct_transform > best_poison_acc):
                best_clean_acc = correct
                best_poison_acc = correct_transform

                print(f'Saving current best model in {test_model_path}')
                if isinstance(netC, torch.nn.DataParallel):
                    torch.save({
                        'atkmodel': atkmodel.module.state_dict(),
                        'netC': netC.module.state_dict(),
                        'optimizerC': optimizerC.state_dict(),
                        'clean_schedulerC': schedulerC,
                        'best_clean_acc': best_clean_acc,
                        'best_poison_acc': best_poison_acc
                    }, test_model_path)
                else:
                    torch.save({
                        'atkmodel': atkmodel.state_dict(),
                        'netC': netC.state_dict(),
                        'optimizerC': optimizerC.state_dict(),
                        'clean_schedulerC': schedulerC,
                        'best_clean_acc': best_clean_acc,
                        'best_poison_acc': best_poison_acc
                    }, test_model_path)

        if cepoch == 1:
            clean_img = data[:args.test_img_row].clone().cpu()
            poison_img = atkdata[:args.test_img_row].clone().cpu()
            residual_p_c = poison_img - clean_img
            residual_c_p = clean_img - poison_img
            noise_original_img = noise_original[:args.test_img_row].clone().cpu()
            noise_process_img = noise_process[:args.test_img_row].clone().cpu()

            clean_img = F.upsample(clean_img, scale_factor=(4, 4))
            poison_img = F.upsample(poison_img, scale_factor=(4, 4))
            residual_p_c = F.upsample(residual_p_c, scale_factor=(4, 4))
            residual_c_p = F.upsample(residual_c_p, scale_factor=(4, 4))
            noise_original_img = F.upsample(noise_original_img, scale_factor=(4, 4))
            noise_process_img = F.upsample(noise_process_img, scale_factor=(4, 4))

            all_img = torch.cat([clean_img, residual_p_c, residual_c_p, noise_original_img, noise_process_img, poison_img], 0)
            grid = torchvision.utils.make_grid(all_img.clone(), nrow=args.test_img_row, normalize=True)
            torchvision.utils.save_image(grid, os.path.join(args.basepath, f'all_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

            torchvision.utils.save_image(torchvision.utils.make_grid(clean_img.clone(), nrow=args.test_img_row, normalize=True),
                                         os.path.join(args.basepath, f'clean_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

            torchvision.utils.save_image(torchvision.utils.make_grid(poison_img.clone(), nrow=args.test_img_row, normalize=True),
                                         os.path.join(args.basepath, f'poison_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

            torchvision.utils.save_image(torchvision.utils.make_grid(residual_p_c.clone(), nrow=args.test_img_row, normalize=True),
                                         os.path.join(args.basepath, f'residual_p_c_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

            torchvision.utils.save_image(torchvision.utils.make_grid(residual_c_p.clone(), nrow=args.test_img_row, normalize=True),
                                         os.path.join(args.basepath, f'residual_c_p_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

            torchvision.utils.save_image(torchvision.utils.make_grid(noise_original_img.clone(), nrow=args.test_img_row, normalize=True),
                                         os.path.join(args.basepath, f'noise_original_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

            torchvision.utils.save_image(torchvision.utils.make_grid(noise_process_img.clone(), nrow=args.test_img_row, normalize=True),
                                         os.path.join(args.basepath, f'noise_process_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

    return clean_accs, poison_accs


def create_table(params):
    data = "\n|" + "name".ljust(30) + "=" + "value".rjust(30) + "| \n" + "-" * 63
    for key, value in sorted(vars(params).items()):
        data += '\n' + "|" + key.ljust(30) + "=" + str(value).rjust(30) + "|"

    return data


def main(args, create_paths, create_models, get_train_test_loaders):

    print('<' * 30 + '>' * 30)
    print('<' * 30 + '>' * 30)
    print('Final Test'.center(60, "="))


    if args.test_alpha is None:
        print(f'使用默认alpha : {args.alpha} 来训练')
        args.test_alpha = args.alpha

    if args.test_lr is None:
        print(f'使用默认学习率 : {args.lr} 来训练')
        args.test_lr = args.lr

    if args.test_eps is None:
        print(f'使用eps : {args.eps / 10} 来训练')
        args.test_eps = args.eps / 10


    args.schedulerC_milestones = [int(e) for e in args.schedulerC_milestones.split(',')]

    # 打印参数
    print('参数如下'.center(61, "="))
    print(create_table(params=args))

    # 确定路径
    args.basepath, args.checkpoint_path, args.bestmodel_path = basepath, checkpoint_path, bestmodel_path = create_paths(args)
    test_model_path = os.path.join(basepath, f'poisoned_classifier_{args.test_alpha}_{args.test_eps}_{args.test_attack_portion}_{args.test_optimizer}_test_model.ckpt')
    # print(f'保存 test model 到 {test_model_path}')

    train_loader, test_loader, clip_image = get_train_test_loaders(args)

    # post_transforms = PostTensorTransform(args).to(args.device)
    # atkmodel, tgtmodel, tgtoptimizer, _, create_net = create_models(args)
    # atkmodel, netC, optimizerC, schedulerC = get_model(args)

    atkmodel, _, netC, _, _ = create_models(args)


    if args.test_use_train_best:

        checkpoint = torch.load(f'{bestmodel_path}')
        if 'clsmodel' in checkpoint and 'atkmodel' in checkpoint:
            print('从 bestmodel_path 加载 atk_checkpoint 和 classifier 参数 : {}'.format(bestmodel_path))
            netC.load_state_dict(checkpoint['clsmodel'], strict=True)
            atk_checkpoint = checkpoint['atkmodel']
        else:
            raise Exception(f"Invalid checkpoint: {bestmodel_path}")

    elif args.test_use_train_last:

        checkpoint = torch.load(f'{checkpoint_path}')
        if 'clsmodel' in checkpoint and 'atkmodel' in checkpoint:
            print('从 checkpoint_path 加载 atk_checkpoint 和 classifier 参数 : {}'.format(checkpoint_path))
            netC.load_state_dict(checkpoint['clsmodel'], strict=True)
            atk_checkpoint = checkpoint['atkmodel']
        else:
            raise Exception(f"Invalid checkpoint: {checkpoint_path}")

    else:

        checkpoint = torch.load(f'{bestmodel_path}')
        if 'atkmodel' in checkpoint:
            print('从 bestmodel_path 加载 atk_checkpoint: {}'.format(bestmodel_path))
            atk_checkpoint = checkpoint['atkmodel']
        else:
            raise Exception(f"Invalid checkpoint: {bestmodel_path}")

    target_transform = get_target_transform(args)

    if args.test_alpha != 1.0:
        print(f'加载 atkmodel')
        atkmodel.load_state_dict(atk_checkpoint, strict=True)
    else:
        print(f'不加载 atkmodel，因为 test_alpha 等于1')

    if args.test_optimizer == 'adam':
        print('使用 adam 优化器')
        optimizerC = torch.optim.Adam(netC.parameters(), args.test_lr, weight_decay=5e-4)

        # Scheduler
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)
    elif args.test_optimizer == 'sgd':
        print('使用 sgd 优化器')
        # optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, weight_decay=5e-4)
        optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, weight_decay=5e-4, momentum=0.9)

        # Scheduler
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)

    if args.use_data_parallel:
        print('使用数据并行')
        netC = torch.nn.DataParallel(netC)
        atkmodel = torch.nn.DataParallel(atkmodel)

    print(netC)
    print(optimizerC)
    print(schedulerC)

    data_transforms = PostTensorTransform(args).to(args.device)
    print('====> Post tensor transform')
    print(data_transforms)

    writer = SummaryWriter(log_dir=basepath)
    clean_accs, poison_accs = final_test(args, test_model_path, atkmodel, netC, target_transform, train_loader, test_loader, trainepoch=args.test_epochs,
                                         writer=writer, log_prefix='POISON', alpha=args.test_alpha, epochs_per_test=1,
                                         optimizerC=optimizerC, schedulerC=schedulerC, data_transforms=data_transforms, clip_image=clip_image)

    print(f'acc_clean : {clean_accs}\tacc_poison : {poison_accs}')


if __name__ == '__main__':
    parser = create_config_parser()
    args = parser.parse_args()
    main(args)
