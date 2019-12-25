from dataloader import SelfPlayLoader
from model import AlphaZero
import numpy as np
import argparse
import torch
from torch import nn


def main(args):
    net = AlphaZero(in_channels=4, layers=args.layers).to(args.device)
    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()
    # TODO: decrease learning rate
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=.0001)
    # dataset
    batch_size = args.batch_size
    print('> Train dataset load from', args.train_dataset)
    train_loader = SelfPlayLoader(args.train_dataset, args.device)
    print('> Test dataset load from', args.test_dataset)
    test_loader = SelfPlayLoader(args.test_dataset, args.device)

    # restore model
    epoch_start = 1
    if args.restore:
        print('> Restore from', args.path)
        checkpoint = torch.load(args.path)
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
        p_loss = p_criterion(p_logits, p_labels.squeeze())
        loss = v_loss * .01 + p_loss
        loss.backward()
        optimizer.step()

        # train accuracy
        with torch.no_grad():
            predicted = torch.argmax(p_logits, dim=1)
            correct = (predicted == p_labels.squeeze()).sum().item()
            print('[{:5d}] Accuracy: {:.2%} Loss: {:.5f}'.format(
                epoch, correct / batch_size, loss.item()))

        if epoch % 1000 == 0:
            # model checkpoint
            torch.save(
                {
                    'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'{args.path}-{epoch}.ckpt')

            # test accuracy
            correct, total = 0, 0
            for features in test_loader:
                inputs, p_labels, _ = zip(*features)
                input_batch = torch.stack(inputs)
                p_batch = torch.stack(p_labels)
                p_logits, _ = net(input_batch)
                predicted = torch.argmax(p_logits, dim=1)
                correct += (predicted == p_batch.squeeze()).sum().item()
                total += len(input_batch)
            print('[{:5d}] Test Accuracy: {:.2%}'.format(
                epoch, correct / total))

    # freeze model
    inputs = torch.rand(1, 4, 9, 9).to(args.device)
    frozen_net = torch.jit.trace(net, inputs)
    frozen_net.save(f'{args.path}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--train-dataset', default='../../self-play')
    parser.add_argument('--test-dataset', default='../../self-play-test')
    # model
    parser.add_argument('-p', '--path', default='model/model')
    parser.add_argument('-r', '--restore', action='store_true')
    # training
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--layers', default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=100000, type=int)
    parser.add_argument('-lr', '--lr', default=.01, type=float)

    args = parser.parse_args()
    main(args)
