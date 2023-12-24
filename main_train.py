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


import iml_vit_model

from engine_train import train_one_epoch, test_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('IML-ViT training', add_help=True)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    #
    parser.add_argument('--vit_pretrain_path', default = '/root/workspace/IML-ViT/pretrained-weights/mae_pretrain_vit_base.pth', type=str, help='path to vit pretrain model by MAE')
    
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--test_period', default=4, type=int,
                        help="how many epoch per testing one time")
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--edge_broaden', default=7, type=int,
                        help='Edge broaden size (in pixels) for edge_generator.')
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='hyper-parameter of the weight for proposed edge loss.')
    parser.add_argument('--predict_head_norm', default="BN", type=str,
                        help="norm for predict head, can be one of 'BN', 'LN' and 'IN' (batch norm, layer norm and instance norm). It may influnce the result  on different machine or datasets!")
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str,
                        help='dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint, input the path of a ckpt.')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def main(args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = utils.iml_transforms.get_albu_transforms('train')
    test_transform = utils.iml_transforms.get_albu_transforms('test')

    # ---- dataset with crop augmentation ----
    if os.path.isdir(args.data_path):
        dataset_train = utils.datasets.mani_dataset(args.data_path, transform=train_transform, edge_width=args.edge_broaden, if_return_shape=True)
    else:
        dataset_train = utils.datasets.json_dataset(args.data_path,transform=train_transform, edge_width = args.edge_broaden, if_return_shape = True)
    
    if os.path.isdir(args.test_data_path):
        dataset_test = utils.datasets.mani_dataset(args.test_data_path, transform=test_transform, edge_width=args.edge_broaden, if_return_shape=True)
    else:
        dataset_test = utils.datasets.json_dataset(args.test_data_path,transform=test_transform, edge_width = args.edge_broaden, if_return_shape = True)

    print(dataset_train)
    print(dataset_test)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

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
    
    # define the model
    model = iml_vit_model.iml_vit_model(
        vit_pretrain_path = args.vit_pretrain_path,
        predict_head_norm= args.predict_head_norm,
        edge_lambda = args.edge_lambda
    )
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    args.opt='AdamW'
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    optimizer  = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_f1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # saving checkpoint
        if args.output_dir and (epoch % 50 == 0 and epoch != 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
        optimizer.zero_grad()
        if epoch  % args.test_period == 0 or epoch + 1 == args.epochs:
            test_stats = test_one_epoch(
                model, 
                data_loader = data_loader_test, 
                device = device, 
                epoch = epoch, 
                log_writer=log_writer,
                args = args
            )
            local_f1 = test_stats['average_f1']
            if local_f1 > best_f1 :
                best_f1 = local_f1
                print("Best F1 = %f" % best_f1)
                if epoch > 35:
                    misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
                
            log_stats =  {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
