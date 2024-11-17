#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:46:30 2023

@author: jwxu
"""

import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import os
import sys
sys.path.append('../../')

from vgg import VGG, mem_VGG
from module import *


parser = argparse.ArgumentParser(description='Memristor-based PyTorch CIFAR10 Training')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=10)
parser.add_argument("--train_batch_size", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--memristor_structure", type=str, default='crossbar') # trace, mimo or crossbar
parser.add_argument("--memristor_device", type=str, default='new_ferro') # ideal, ferro, or hu
parser.add_argument("--c2c_variation", type=bool, default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=int, default=0) # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
parser.add_argument("--input_bit", type=int, default=8)
parser.add_argument("--ADC_precision", type=int, default=16)
parser.add_argument("--ADC_setting", type=int, default=4)  # 2:two memristor crossbars use one ADC; 4:one memristor crossbar use one ADC
parser.add_argument("--ADC_rounding_function", type=str, default='floor')  # floor or round
parser.add_argument("--wire_width", type=int, default=200) # In practice, wire_width shall be set around 1/2 of the memristor size; Hu: 10um; Ferro:200nm;
parser.add_argument("--CMOS_technode", type=int, default=32)
parser.add_argument("--device_roadmap", type=str, default='HP')  # HP: High Performance or LP: Low Power
parser.add_argument("--temperature", type=int, default=300)
parser.add_argument("--hardware_estimation", type=int, default=True)
args = parser.parse_args()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Memristor write
        for layer in net.features.children():
            if isinstance(layer, Mem_Conv2d):
                if args.stuck_at_fault == True:
                    layer.crossbar.update_SAF_mask()
                layer.mem_update()
        if isinstance(net.classifier, Mem_Linear):
            if args.stuck_at_fault == True:
                net.classifier.crossbar.update_SAF_mask()
            net.classifier.mem_update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, '/', len(trainloader), ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Accuracy Results:' + str(100. * correct / total) + '\n')

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/mem_ckpt.pth')
        best_acc = acc


if __name__ == '__main__':
    # Sets up Gpu use
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
    seed = args.seed
    gpu = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False
    print("Running on Device = ", device)

    # Mem device setup
    sim_params = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
                  'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
                  'stuck_at_fault': args.stuck_at_fault, 'retention_loss': args.retention_loss,
                  'aging_effect': args.aging_effect, 'wire_width': args.wire_width, 'input_bit': args.input_bit,
                  'batch_interval': 1, 'CMOS_technode': args.CMOS_technode, 'ADC_precision': args.ADC_precision,
                  'ADC_setting': args.ADC_setting, 'ADC_rounding_function': args.ADC_rounding_function,
                  'device_roadmap': args.device_roadmap, 'temperature': args.temperature,
                  'hardware_estimation': args.hardware_estimation}

    best_acc = 0
    start_epoch = 0

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = mem_VGG('VGG16', mem_device=sim_params)
    net = net.to(device)

    # print area results
    if sim_params['hardware_estimation']:
        total_area = 0
        for layer in net.features.children():
            if isinstance(layer, Mem_Conv2d):
                layer.crossbar.total_area_calculation()
                sim_area = layer.crossbar.sim_area
                total_area += sim_area['sim_total_area']
        if isinstance(net.classifier, Mem_Linear):
            layer = net.classifier
            layer.crossbar.total_area_calculation()
            sim_area = layer.crossbar.sim_area
            total_area += sim_area['sim_total_area']
        print("total_area=" + str(total_area))

    # Memristor write
    for layer in net.features.children():
        if isinstance(layer, Mem_Conv2d):
            if args.stuck_at_fault == True:
                layer.crossbar.update_SAF_mask()
            layer.mem_update()
    if isinstance(net.classifier, Mem_Linear):
        if args.stuck_at_fault == True:
            net.classifier.crossbar.update_SAF_mask()
        net.classifier.mem_update()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)
        scheduler.step()

