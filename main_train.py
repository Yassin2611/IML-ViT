# train.py

# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import utils.datasets
import utils.iml_transforms
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

# Replace the old import with the Wvlet model
from iml_vit_model_wvlet import IMLViT_Wvlet

from engine_train import train_one_epoch, test_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('IML-ViT training', add_help=True)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")

    # Model architecture switches
    parser.add_argument('--input_size',        default=1024,    type=int,
                        help='input image size')
    parser.add_argument('--patch_size',        default=16,      type=int,
                        help='patch size for ViT')
    parser.add_argument('--embed_dim',         default=768,     type=int,
                        help='embedding dim for ViT')
    parser.add_argument('--wvlet_levels',      default=[4,2,1,0.5], nargs='+', type=float,
                        help='wavelet tree levels (list of scale factors)')
    parser.add_argument('--wavelet',           default='db1',   type=str,
                        help='wavelet type for WvletTree')
    parser.add_argument('--mlp_embedding_dim', default=256,     type=int,
                        help='MLP embedding dim for PredictHead')

    parser.add_argument('--vit_pretrain_path', default=None, type=str,
                        help='path to ViT pretrained checkpoint (MAE format)')

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--test_period', default=4, type=int,
                        help="how many epochs between evaluation")
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--edge_broaden', default=7, type=int,
                        help='edge broaden size (pixels)')
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='weight for edge loss')
    parser.add_argument('--predict_head_norm', default="BN", type=str,
                        help="norm for predict head (BN, LN, IN)")

    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='absolute learning rate')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base lr: lr = blr * batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N')

    # Data
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--test_data_path', default=None, type=str)
    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--log_dir', default='./output_dir', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)

    return parser


def main(args):
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = utils.iml_transforms.get_albu_transforms('train')
    test_transform  = utils.iml_transforms.get_albu_transforms('test')

    # Dataset
    if os.path.isdir(args.data_path):
        dataset_train = utils.datasets.mani_dataset(
            args.data_path, transform=train_transform,
            edge_width=args.edge_broaden, if_return_shape=True)
    else:
        dataset_train = utils.datasets.json_dataset(
            args.data_path, transform=train_transform,
            edge_width=args.edge_broaden, if_return_shape=True)

    if os.path.isdir(args.test_data_path):
        dataset_test = utils.datasets.mani_dataset(
            args.test_data_path, transform=test_transform,
            edge_width=args.edge_broaden, if_return_shape=True)
    else:
        dataset_test = utils.datasets.json_dataset(
            args.test_data_path, transform=test_transform,
            edge_width=args.edge_broaden, if_return_shape=True)

    print(dataset_train)
    print(dataset_test)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(),
            rank=misc.get_rank(), shuffle=True)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=misc.get_world_size(),
            rank=misc.get_rank(), shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test  = torch.utils.data.SequentialSampler(dataset_test)

    log_writer = None
    if misc.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ----------------------------
    # Build the model
    # ----------------------------
    model = IMLViT_Wvlet(
        input_size        = args.input_size,
        patch_size        = args.patch_size,
        embed_dim         = args.embed_dim,
        vit_pretrain_path = args.vit_pretrain_path,
        wvlet_levels      = args.wvlet_levels,
        wavelet           = args.wavelet,
        mlp_embedding_dim = args.mlp_embedding_dim,
        predict_head_norm = args.predict_head_norm,
        edge_lambda       = args.edge_lambda
    )

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model_without_ddp = model

    print("Model =\n", model_without_ddp)

    # learning rate scaling
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"actual lr: {args.lr:.2e}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=False)
        model_without_ddp = model.module

    # Optimizer & scaler
    optimizer   = optim_factory.create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_f1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch,
            loss_scaler=loss_scaler, log_writer=log_writer, args=args)

        # save checkpoint every 50 epochs
        if args.output_dir and (epoch % 50 == 0 and epoch > 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # evaluation
        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            test_stats = test_one_epoch(
                model, data_loader_test, device, epoch,
                log_writer=log_writer, args=args)
            local_f1 = test_stats['average_f1']
            if local_f1 > best_f1 and epoch > 35:
                best_f1 = local_f1
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            print(f"Best F1 so far: {best_f1:.4f}")
        else:
            test_stats = {}

        # logging
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time', str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
