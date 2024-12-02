import argparse
import os
import time
import sys

sys.path.append('../')

from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset
import mlp
from module import *
from Memristor_Modeling.full_fitting_flow import full_fitting

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=101, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--data_root', default='data/', help='folder to save the model')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
parser.add_argument("--memristor_structure", type=str, default='crossbar') # crossbar
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'train_log')
print = misc.logger.info

# %% Obtain memristor parameters
sim_params = full_fitting(args.memristor_structure, None)

# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

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
print("Running on Device = " + str(device))

# data loader
train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)

# %% Multiple test
out_root = 'Accuracy_Results_MLP_train.txt'
multiple_test_no = 50
for test_cnt in range(multiple_test_no):
    # model
    # model = mlp.mlp_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=False)
    model = mlp.mem_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=False, mem_device=sim_params)
    model.to(device)

    # Area print
    if sim_params['hardware_estimation']:
        total_area = 0
        for layer_name, layer in model.layers.items():
            if isinstance(layer, Mem_Linear):
                layer.crossbar.total_area_calculation()
                total_area += layer.crossbar.sim_area['sim_total_area']
        print("total area=" + str(total_area) + " m2")

    # Memristor write
    for layer_name, layer in model.layers.items():
        if isinstance(layer, Mem_Linear):
            layer.mem_update()
            if sim_params['stuck_at_fault'] == True:
                layer.crossbar.update_SAF_mask()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))
    best_acc, old_file = 0, None
    best_epoch = None
    t_begin = time.time()
    try:
        # ready to go
        for epoch in range(args.epochs):
            model.train()
            if epoch in decreasing_lr:
                optimizer.param_groups[0]['lr'] *= 0.1
            for batch_idx, (data, target) in enumerate(train_loader):
                indx_target = target.clone()
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                # Memristor write
                for layer_name, layer in model.layers.items():
                    if isinstance(layer, Mem_Linear):
                        layer.mem_update()
                        if sim_params['stuck_at_fault'] == True:
                            layer.crossbar.update_SAF_mask()

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct = pred.cpu().eq(indx_target).sum()
                    acc = correct * 1.0 / len(data)
                    print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        loss.data, acc, optimizer.param_groups[0]['lr']))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time
            print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapse_time, speed_epoch, speed_batch, eta))
            misc.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

            if sim_params['hardware_estimation']:
                # print power results
                total_energy = 0
                average_power = 0
                total_read_energy = 0
                total_write_energy = 0
                total_reset_energy = 0
                for layer_name, layer in model.layers.items():
                    if isinstance(layer, Mem_Linear):
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

            if epoch % args.test_interval == 0:
                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(test_loader):
                        indx_target = target.clone()
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.cpu().eq(indx_target).sum()

                test_loss = test_loss / len(test_loader)  # average over number of mini-batch
                acc = 100. * correct / len(test_loader.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(test_loader.dataset), acc))
                if acc > best_acc:
                    new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                    misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    best_epoch = epoch
                    old_file = new_file

    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
        out = open(out_root, 'a')
        out_txt = 'Best accuracy:' + str(best_acc) + '\tbest epoch:' + str(best_epoch) + '\ttest epoch:' + str(
            epoch) + '\ttotal time:' + str(time.time() - t_begin) + '\n'
        out.write(out_txt)
        out.close()