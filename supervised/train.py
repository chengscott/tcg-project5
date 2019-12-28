from dataloader import SelfPlayLoader
from model import AlphaZero
import numpy as np
import argparse
import torch
from torch import nn


def main(args):
    net = AlphaZero(in_channels=4, layers=args.layers).to(args.device)
    p_criterion = lambda p_logits, p_labels: (
        (-p_labels * torch.log_softmax(p_logits, dim=1)).sum(dim=1).mean())
    v_criterion = nn.MSELoss()
    v_weight = .1 if args.supervised else 1.
    # TODO: decrease learning rate
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=.0001)
    # dataset
    batch_size = args.batch_size
    print('> Dataset load from', args.dataset)
    train_loader = SelfPlayLoader(args.dataset,
                                  args.device,
                                  load_policy=not args.supervised)

    # restore model
    epoch_start = 1
    if args.restore:
        print('> Restore from', args.load)
        checkpoint = torch.load(args.load)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        print('> Start from [{:6d}]'.format(epoch_start))

    for epoch in range(epoch_start, args.epochs + 1):
        inputs, p_labels, v_labels = train_loader.sample(batch_size)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        p_logits, v_logits = net(inputs)
        v_loss = v_criterion(v_logits, v_labels)
        p_loss = p_criterion(p_logits, p_labels)
        loss = v_loss * v_weight + p_loss
        loss.backward()
        optimizer.step()

        # train loss
        with torch.no_grad():
            print('[{:5d}] PN_Loss: {:.5f} VN_Loss: {:.5f}'.format(
                epoch, p_loss.item(), v_loss.item()))

        if epoch % 10000 == 0:
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
    # training
    parser.add_argument('--dataset', default='../../self-play')
    parser.add_argument('--supervised', action='store_true')
    # network
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--layers', default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=100000, type=int)
    parser.add_argument('-lr', '--lr', default=.01, type=float)

    args = parser.parse_args()
    main(args)
