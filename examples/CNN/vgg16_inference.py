#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:46:30 2023

@author: jwxu
"""

import argparse
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

import os
import sys
sys.path.append('../../')

from vgg import VGG, mem_VGG
from module import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=25)
parser.add_argument("--memristor_structure", type=str, default='crossbar') # trace, mimo or crossbar
parser.add_argument("--memristor_device", type=str, default='ferro') # ideal, ferro, or hu
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
parser.add_argument("--device_roadmap", type=str, default='HP') # HP: High Performance or LP: Low Power
parser.add_argument("--temperature", type=int, default=300)
parser.add_argument("--hardware_estimation", type=int, default=True)
args = parser.parse_args()

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
              'ADC_setting': args.ADC_setting,'ADC_rounding_function': args.ADC_rounding_function,
              'device_roadmap': args.device_roadmap, 'temperature': args.temperature,
              'hardware_estimation': args.hardware_estimation}

# Dataset prepare
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Network Model
print('==> Building memristor-based model..')
# net = VGG('VGG16')
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

# Load Pre-trained Model
checkpoint = torch.load('./checkpoint/ckpt.pth')
recorded_best_acc = checkpoint['acc']

# Repeated Experiment
out_root = 'Inference_results.txt'
for test_cnt in range(args.rep):
    # Reset Dataset
    testloader.idx = 0
    
    # Record
    out = open(out_root, 'a')

    # Reload Weight
    state_dict = checkpoint['net']
    # State_dict adaption
    adapt_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        match_str = [s for s in state_dict.keys() if k in s]
        if len(match_str) > 0:
            adapt_state_dict[k] = state_dict[match_str[0]]
        else:
            adapt_state_dict[k] = v
    net.load_state_dict(adapt_state_dict)

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


    net = net.to(device)

    # Evaluate
    print('==> Evaluate..')
    criterion = nn.CrossEntropyLoss()
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
    
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    print('Accuracy Results:' + str(acc))

    if sim_params['hardware_estimation']:
        # print power results
        total_energy = 0
        average_power = 0
        total_read_energy = 0
        total_write_energy = 0
        total_reset_energy = 0
        for layer in net.features.children():
            if isinstance(layer, Mem_Conv2d):
                layer.crossbar.total_energy_calculation()
                sim_power = layer.crossbar.sim_power
                total_read_energy += sim_power['read_energy']
                total_write_energy += sim_power['write_energy']
                total_reset_energy += sim_power['reset_energy']
                total_energy += sim_power['total_energy']
                average_power += sim_power['average_power']
        if isinstance(net.classifier, Mem_Linear):
            layer = net.classifier
            layer.crossbar.total_energy_calculation()
            sim_power = layer.crossbar.sim_power
            total_read_energy += sim_power['read_energy']
            total_write_energy += sim_power['write_energy']
            total_reset_energy += sim_power['reset_energy']
            total_energy += sim_power['total_energy']
            average_power += sim_power['average_power']

        print("total_energy=" + str(total_energy))
        print("total_read_energy=" + str(total_read_energy))
        print("total_write_energy=" + str(total_write_energy))
        print("total_reset_energy=" + str(total_reset_energy))
        print("average_power=" + str(average_power))
    
    out_txt = 'Accuracy:' + str(acc) + '\n'
    out.write(out_txt)
    out.close()