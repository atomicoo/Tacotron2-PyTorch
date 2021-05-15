import argparse

import numpy as np
import torch
# from torch import nn
import torch.optim as optim
from tqdm import tqdm

import config
from datasets import Text2MelDataset, Text2MelDataLoader
from models import Tacotron2
from models.losses import Tacotron2Loss, AverageMeter
from models.optims import Tacotron2Optimizer
from helpers.logger import Logger
from utils.common import save_checkpoint, load_checkpoint


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logdir = args.logdir
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        model = Tacotron2(config)
        # optimizer
        optimizer = Tacotron2Optimizer(
            optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.9, 0.999), eps=1e-6))

    else:
        start_epoch, epochs_since_improvement, model, optimizer, best_loss = load_checkpoint(logdir, checkpoint)

    logger = Logger(config.logdir, config.experiment, 'tacotron2')

    # Move to GPU, if available
    model = model.to(config.device)

    criterion = Tacotron2Loss()

    # Custom dataloaders
    train_dataset = Text2MelDataset(config.train_files, config)
    train_loader = Text2MelDataLoader(train_dataset, config, shuffle=True, 
                                      num_workers=args.num_workers, pin_memory=True)
    valid_dataset = Text2MelDataset(config.valid_files, config)
    valid_loader = Text2MelDataLoader(valid_dataset, config, shuffle=False, 
                                      num_workers=args.num_workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           epoch=epoch,
                           logger=logger)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        scalar_dict = { 'train_epoch_loss': train_loss, 
                        'learning_rate': lr }
        logger.log_epoch('train', epoch, scalar_dict=scalar_dict)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterion=criterion,
                           logger=logger)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: {}\n".format(epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        scalar_dict = { 'valid_epoch_loss': valid_loss }
        logger.log_epoch('valid', epoch, scalar_dict=scalar_dict)

        # Save checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(logdir, epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)

        # # alignments
        # img_align = test(model, optimizer.step_num, valid_loss)
        # writer.add_image('model/alignment', img_align, epoch, dataformats='HWC')


def train(train_loader, model, optimizer, criterion, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, batch in enumerate(train_loader):
        model.zero_grad()
        x, y = model.parse_batch(batch)

        # Forward prop.
        y_pred = model(x)

        loss = criterion(y_pred, y)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

            scalar_dict = { 'train_step_loss': loss.item() }
            logger.log_step('train', optimizer.step_num, scalar_dict)

    return losses.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for batch in tqdm(valid_loader):
        model.zero_grad()
        x, y = model.parse_batch(batch)

        # Forward prop.
        y_pred = model(x)

        loss = criterion(y_pred, y)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    print('\nValid Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg


def parse_args():
    parser = argparse.ArgumentParser(description='Tacotron2')
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=float, help='Gradient norm threshold to clip')
    # minibatch
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to generate minibatch')
    # logging
    parser.add_argument('--logdir', default='logdir', type=str, help='Logging directory')
    parser.add_argument('--print_freq', default=1, type=int, help='Frequency of printing training information')
    parser.add_argument('--save_freq', default=1, type=int, help='Frequency of saving model checkpoint')
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
    parser.add_argument('--l2', default=1e-6, type=float, help='weight decay (L2)')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    # others
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    args = parser.parse_args()
    return args


def main():
    global args
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
