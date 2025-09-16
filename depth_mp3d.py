import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn
from mp3d_dataset import load_MP3D_data
from model_mp3d import ModelMP3D

import numpy as np
import os
import argparse
import time

def train(net, args, device):

    time_start = time.time()
    for epoch in range(args.resume, args.epochs):
        net.train()

        trainloader = load_MP3D_data(args.batch_size, stage='train')

        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))

        for Loop, (rgb_paths) in enumerate(trainloader, 0):
            rgb_paths_list = [element
                for item in rgb_paths
                for element in (item if isinstance(item, tuple) else (item,))
            ]
            # print(rgb_paths)
            net.create_data(rgb_paths_list)

            if Loop % 10 == 9:  #
                time_end = time.time()

                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                        ' Time: ' + str(time_end - time_start))

                time_start = time_end

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--train', type=int, default=1, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    parser.add_argument('--area', type=str, default='same', help='same or cross')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    net = ModelMP3D(args, device)
    net.to(device)

    train(net, args, device)

