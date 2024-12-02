import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import os
import sys
sys.path.append('../../')
sys.path.append('../')

from vgg import VGG, mem_VGG
from module import *
from Memristor_Modeling.full_fitting_flow import full_fitting

parser = argparse.ArgumentParser(description='Memristor-based PyTorch CIFAR10 Training')
# network configuration
parser.add_argument("--seed", type=int, default=0) # Random seed
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=10) # Number of repetitions for the experiment
parser.add_argument("--train_batch_size", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# circuit configuration
parser.add_argument("--memristor_structure", type=str, default='crossbar') # crossbar
args = parser.parse_args()

# %% Obtain memristor parameters
sim_params = full_fitting(args.memristor_structure, None)

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
                layer.mem_update()
                if sim_params['stuck_at_fault'] == True:
                    layer.crossbar.update_SAF_mask()
        if isinstance(net.classifier, Mem_Linear):
            net.classifier.mem_update()
            if sim_params['stuck_at_fault'] == True:
                net.classifier.crossbar.update_SAF_mask()

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
            layer.mem_update()
            if sim_params['stuck_at_fault'] == True:
                layer.crossbar.update_SAF_mask()
    if isinstance(net.classifier, Mem_Linear):
        net.classifier.mem_update()
        if sim_params['stuck_at_fault'] == True:
            net.classifier.crossbar.update_SAF_mask()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)
        scheduler.step()

