# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched

from utils.datasets import denormalize
import utils.evaluation as evaluation

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
            
    for data_iter_step, (samples, masks, edge_mask, shape) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
                
        samples, masks, edge_mask = samples.to(device), masks.to(device), edge_mask.to(device)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        torch.cuda.synchronize()
        
        with torch.cuda.amp.autocast():
            predict_loss, predict, edge_loss = model(samples, masks, edge_mask, shape)
            predict_loss_value = predict_loss.item()
            edge_loss_value = edge_loss.item()
            
        predict_loss = predict_loss / accum_iter
        loss_scaler(predict_loss,optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        # save to log.txt
        metric_logger.update(lr=lr)
        metric_logger.update(predict_loss= predict_loss_value)
        metric_logger.update(edge_loss= edge_loss_value)
        loss_predict_reduce = misc.all_reduce_mean(predict_loss_value)
        edge_loss_reduce = misc.all_reduce_mean(edge_loss_value)

        if log_writer is not None and (data_iter_step + 1) % 50 == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # Tensorboard logging
            log_writer.add_scalar('train_loss/predict_loss', loss_predict_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss/edge_loss', edge_loss_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
               
    if log_writer is not None:
        log_writer.add_images('train/image',  denormalize(samples), epoch)
        log_writer.add_images('train/predict', predict, epoch)
        log_writer.add_images('train/predict_t', (predict > 0.5) * 1.0, epoch)
        log_writer.add_images('train/masks', masks, epoch)
        log_writer.add_images('train/edge_mask', edge_mask, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    epoch: int, 
                    log_writer=None,
                    args=None):
    with torch.no_grad():
        model.zero_grad()
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        # F1 evaluation for an Epoch during training
        print_freq = 20
        header = 'Test: [{}]'.format(epoch)
        for data_iter_step, (images, masks, edge_mask, shape) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
            images, masks, edge_mask = images.to(device), masks.to(device), edge_mask.to(device)
            predict_loss, predict, edge_loss = model(images, masks, edge_mask)
            predict = predict.detach()
            #---- Training evaluation ----
            # region_mask is for cutting of the zero-padding area.
            region_mask = evaluation.genertate_region_mask(masks, shape) 
            TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict, masks, region_mask)
        
            local_f1 = evaluation.cal_F1(TP, TN, FP, FN)
            # print(local_f1)
            
            for i in local_f1: # merge batch
                metric_logger.update(average_f1=i)
                print(metric_logger.meters['average_f1'].count)
                print(metric_logger.meters['average_f1'].total)

        metric_logger.synchronize_between_processes()    
        # print("---syncronized---")
        # print(metric_logger.meters['average_f1'].count)
        # print(metric_logger.meters['average_f1'].total)
        # print('---syncronized done ---')
        if log_writer is not None:
            log_writer.add_scalar('F1/test_average', metric_logger.meters['average_f1'].global_avg, epoch)
            log_writer.add_images('test/image',  denormalize(images), epoch)
            log_writer.add_images('test/predict', (predict > 0.5)* 1.0, epoch)
            log_writer.add_images('test/masks', masks, epoch)
            log_writer.add_images('test/edge_mask', edge_mask, epoch)
            
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}