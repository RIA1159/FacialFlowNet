from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from network import RAFTGMA

from utils import flow_viz
import datasets
import evaluate

from torch.cuda.amp import GradScaler

# exclude extremly large displacements
MAX_FLOW = 400


def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(facial_pred, head_pred, flow_gt_exp, flow_gt_head,  valid_e, valid_h, gamma):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(facial_pred)    
    flow_loss = 0.0

    facial_loss = 0.0
    valid_e = (valid_e >= 0.5) & ((flow_gt_exp**2).sum(dim=1).sqrt() < MAX_FLOW)


    head_loss = 0.0
    valid_h = (valid_h >= 0.5) & ((flow_gt_head**2).sum(dim=1).sqrt() < MAX_FLOW)


    exp_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (facial_pred[i] - flow_gt_exp).abs()
        facial_loss += i_weight * (valid_e[:, None] * i_loss).mean()

        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (head_pred[i] - flow_gt_head).abs()
        head_loss += i_weight * (valid_h[:, None] * i_loss).mean()


        i_weight = gamma**(n_predictions - i - 1)
        i_loss = ((facial_pred[i] - head_pred[i]) - (flow_gt_exp - flow_gt_head)).abs()
        exp_loss += i_weight * (valid_h[:, None] * i_loss).mean()




    flow_loss = facial_loss + 0.8*head_loss +  1.5*exp_loss


    epe = torch.sum((head_pred[-1] - flow_gt_head)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid_e.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0 + PRETRAINED
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.uint8)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}


def main(args):

    model = nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)
    
    print(f"Parameter Count: {count_parameters(model)}")
    

    pre_trained = torch.load(args.restore_ckpt)

    if args.restore_ckpt is not None:
        model.load_state_dict(pre_trained, strict=False)




    model.cuda()
    model.train()


    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)
            plot_val(logger, args)
            break

    PATH = args.output+f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    return PATH


def flow2image(flow):
    flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    image = flow_viz.flow_to_image(flow)

    return image

def vis_training(facial_pred, head_pred,   flow_e, flow_h):

    pred_head_flow = head_pred[-1]
    pred_facial_flow = facial_pred[-1]
    pred_exp_flow = pred_facial_flow - pred_head_flow


    gt_facial_flow = flow_e
    gt_head_flow = flow_h
    gt_exp_flow = flow_e - flow_h

    flow_list = [pred_facial_flow, pred_head_flow, pred_exp_flow, gt_facial_flow, gt_head_flow, gt_exp_flow]

    for i, flow in enumerate(flow_list):
        if i == 0:
            result = flow2image(flow)
        else:
            image = flow2image(flow)
            
            result = np.concatenate((result, image), axis = 1)

    cv2.imwrite("./results.jpg", result)

def train(model, train_loader, optimizer, scheduler, logger, scaler, args):




    for i_batch, data_blob in enumerate(train_loader):

        tic = time.time()
   
        image1, image2, flow_e, flow_h, valid_e, valid_h, mask = [x.cuda() for x in data_blob]


        optimizer.zero_grad()

        facial_pred, head_pred = model(image1, image2)

        loss, metrics = sequence_loss(facial_pred, head_pred, flow_e, flow_h, valid_e, valid_h, args.gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()

        metrics['time'] = toc - tic
        logger.push(metrics)

        # if logger.total_steps % 100 == 0:
        #     vis_training(facial_pred, head_pred, flow_e, flow_h)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:

            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            torch.save(model.state_dict(), PATH)

            validate(model, args, logger)
            plot_train(logger, args)
            plot_val(logger, args)
            

        if logger.total_steps >= args.num_steps:
            break


def validate(model, args, logger):
    model.eval()
    results = {}

    # Evaluate results
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            results.update(evaluate.validate_chairs(model.module, args.iters))
        elif val_dataset == 'sintel':
            results.update(evaluate.validate_sintel(model.module, args.iters))
        elif val_dataset == 'kitti':
            results.update(evaluate.validate_kitti(model.module, args.iters))


        elif val_dataset == 'facialflow':
            validate_result = evaluate.validate_facialflow(model.module)
            results.update(validate_result)

            print(validate_result['exps-epe'])
            with open("./evalute_result.txt", "a+") as f:
                f.write("epoch: {} epe: {}\n".format(logger.total_steps + 1 , validate_result['exps-epe']))
                f.close()
        


            
    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    logger.val_steps_list.append(logger.total_steps)
    model.train()


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints1', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--frozen', default=False, action='store_true',
                        help='frozen facialflow head')


    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=100,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    PRETRAINED = 0

    main(args)
