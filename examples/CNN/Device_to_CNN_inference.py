import argparse
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

import os
import time
import sys
sys.path.append('../../')
sys.path.append('../')


from vgg import VGG, mem_VGG
from module import *
from Memristor_Modeling.full_fitting_flow import full_fitting

parser = argparse.ArgumentParser()
# network configuration
parser.add_argument("--seed", type=int, default=0) # Random seed
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=10) # Number of repetitions for the experiment
parser.add_argument("--batch_size", type=int, default=25) # Batch size for data loading
# circuit configuration
parser.add_argument("--memristor_structure", type=str, default='crossbar') # crossbar
args = parser.parse_args()

# Sets up Gpu use
seed = args.seed
# seed = int(time.time()) # Random Seed
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

# %% Obtain memristor parameters
sim_params = full_fitting(args.memristor_structure, None)

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

# Repeated Experiment
out_root = 'Inference_results.txt'
for test_cnt in range(args.rep):
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
            layer.mem_update()
            if sim_params['stuck_at_fault'] == True:
                layer.crossbar.update_SAF_mask()
    if isinstance(net.classifier, Mem_Linear):
        net.classifier.mem_update()
        if sim_params['stuck_at_fault'] == True:
            net.classifier.crossbar.update_SAF_mask()


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