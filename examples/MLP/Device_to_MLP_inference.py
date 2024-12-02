import argparse
import os
import time
import sys

sys.path.append('../')

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset
import mlp
from module import *
from Memristor_Modeling.full_fitting_flow import full_fitting

parser = argparse.ArgumentParser()
# network configuration
parser.add_argument("--seed", type=int, default=0) # Random seed
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=10) # Number of repetitions for the experiment
parser.add_argument("--batch_size", type=int, default=100) # Batch size for data loading
parser.add_argument('--data_root', default='data/', help='folder to save the model')
# circuit configuration
parser.add_argument("--memristor_structure", type=str, default='crossbar') #crossbar
args = parser.parse_args()

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

# %% Obtain memristor parameters
sim_params = full_fitting(args.memristor_structure, None)

t_begin = time.time()

# Dataset prepare
print('==> Preparing data..')
test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1, train=False, val=True)

# Repeated Experiment
print('==> Read Memristor..')
out_root = 'MLP_inference_results.txt'

for test_cnt in range(args.rep):
    # Network Model
    # model = mlp.mlp_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=True,)
    model = mlp.mem_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=True, mem_device=sim_params)

    # Area print
    if sim_params['hardware_estimation']:
        total_area = 0
        for layer_name, layer in model.layers.items():
            if isinstance(layer, Mem_Linear):
                layer.crossbar.total_area_calculation()
                total_area += layer.crossbar.sim_area['sim_total_area']
        print("total area=", total_area, " m2")

    # Memristor write
    print('==> Write Memristor..')
    start_time = time.time()
    for layer_name, layer in model.layers.items():
        if isinstance(layer, Mem_Linear):
            layer.mem_update()
            if sim_params['stuck_at_fault'] == True:
                layer.crossbar.update_SAF_mask()
    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)

    if sim_params['hardware_estimation']:
        # print write power results
        total_energy = 0
        average_power = 0
        for layer_name, layer in model.layers.items():
            if isinstance(layer, Mem_Linear):
                layer.crossbar.total_energy_calculation()
                sim_power = layer.crossbar.sim_power
                total_energy += sim_power['total_energy']
                average_power += sim_power['average_power']
        print("\ttotal_write_energy=", total_energy)
        print("\taverage_write_power=", average_power)

    model.to(device)

    # Reset Dataset
    test_loader.idx = 0

    # Record
    out = open(out_root, 'a')

    # Evaluate
    print('==> Evaluate..')
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

    acc = 100. * correct / len(test_loader.dataset)
    print('\tTest Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), acc))

    if sim_params['hardware_estimation']:
        # print power results
        total_energy = 0
        average_power = 0
        for layer_name, layer in model.layers.items():
            if isinstance(layer, Mem_Linear):
                layer.crossbar.total_energy_calculation()
                sim_power = layer.crossbar.sim_power
                total_energy += sim_power['total_energy']
                average_power += sim_power['average_power']
        print("\ttotal_energy=", total_energy)
        print("\taverage_power=", average_power)

    out_txt = 'Accuracy:' + str(acc) + '\n'
    out.write(out_txt)
    out.close()

elapse_time = time.time() - t_begin
print("Total Elapse: {:.2f}".format(time.time() - t_begin))