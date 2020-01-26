from dataloader_nfs import SupervisedShiftHelper as SupervisedHelper
from model import AlphaZero
import numpy as np
import argparse
import torch
from torch import nn
import time


def main(args):
    net = AlphaZero(in_channels=4,
                    layers=args.layers,
                    channels=args.channels,
                    bias=True).to(args.device)
    p_criterion = lambda p_logits, p_labels: (
        (-p_labels * torch.log_softmax(p_logits, dim=1)).sum(dim=1).mean())
    v_criterion = nn.MSELoss()
    # TODO: decay learning rate
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=.0001,
                                nesterov=True)
    # dataset
    batch_size = args.batch_size
    datahelper = SupervisedHelper(args.dataset, args.dataset_prefix,
                                  args.device)

    # restore model
    if args.restore:
        print('> Restore from', args.load)
        checkpoint = torch.load(args.load)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #datahelper.init_epoch(checkpoint['epoch'])
        #print('> Start from [{:6d}]'.format(epoch_start))

    for epoch, dname, num_samples, dataloader in datahelper:
        print('> Dataset load from', dname)
        num_samples = num_samples // batch_size
        time_start = time.time()
        for i in range(num_samples):
            inputs, p_labels, v_labels = dataloader.sample(batch_size)

            # forward + backward
            p_logits, v_logits = net(inputs)
            v_loss = v_criterion(v_logits, v_labels)
            p_loss = p_criterion(p_logits, p_labels)
            loss = (v_loss + p_loss).mean()

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train loss
            with torch.no_grad():
                print('[{:3d}:{:5d}] PN_Loss: {:.5f} VN_Loss: {:.5f}'.format(
                    epoch, i, p_loss.item(), v_loss.item()))

        print('Time:', time.time() - time_start)
        # model checkpoint
        torch.save(
            {
                'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'{args.path}-{epoch}.ckpt')

    # freeze model
    inputs = torch.rand(1, 4, 9, 9).to(args.device)
    frozen_net = torch.jit.trace(net, inputs)
    frozen_net.save(f'{args.path}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-p', '--path', default='model/model')
    parser.add_argument('-l', '--load', default='model/model.ckpt')
    parser.add_argument('-r', '--restore', action='store_true')
    # dataset
    parser.add_argument('--dataset',
                        default='../../self-play-nfs/f128_10_record')
    parser.add_argument('--dataset-prefix',
                        default='/mnt/nfs_share/NoGo/f128_10/record/')
    # network
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--layers', default=10, type=int)
    parser.add_argument('--channels', default=128, type=int)
    parser.add_argument('-bs', '--batch_size', default=1024, type=int)
    parser.add_argument('-lr', '--lr', default=.01, type=float)

    args = parser.parse_args()
    main(args)
