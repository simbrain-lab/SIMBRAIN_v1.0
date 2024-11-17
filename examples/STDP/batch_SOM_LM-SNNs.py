#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:49:32 2022

@author: zhangxin
"""

import os
import torch
import argparse
import numpy as np
import math

import sys
sys.path.append('../../')

from torchvision import transforms
from tqdm import tqdm
from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels


# %% Argument
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=625)
parser.add_argument("--train_batch_size", type=int, default=50)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--update_inhibation_weights", type=int, default=500)
parser.add_argument("--plot_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--memristor_structure", type=str, default='trace') # trace or crossbar 
parser.add_argument("--memristor_device", type=str, default='trace') # trace: original trace
parser.add_argument("--c2c_variation", type=bool, default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=int, default=0) # 0: No retention, 1: during pulse, 2: no pluse for a long time
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
parser.add_argument("--input_bit", type=int, default=1)
parser.add_argument("--ADC_precision", type=int, default=8)
parser.add_argument("--ADC_setting", type=int, default=4)  # 2:two memristor crossbars use one ADC; 4:one memristor crossbar use one ADC
parser.add_argument("--ADC_rounding_function", type=str, default='floor')  # floor or round
parser.add_argument("--wire_width", type=int, default=200) # In practice, wire_width shall be set around 1/2 of the memristor size; Hu: 10um; Ferro:200nm;
parser.add_argument("--CMOS_technode", type=int, default=32)
parser.add_argument("--device_roadmap", type=str, default='HP') # HP: High Performance or LP: Low Power
parser.add_argument("--temperature", type=int, default=300)
parser.add_argument("--hardware_estimation", type=int, default=False)

parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
n_epochs = args.n_epochs
n_test = math.ceil(args.n_test / test_batch_size)
n_train = math.ceil(args.n_train / train_batch_size)
n_workers = args.n_workers
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
plot_interval = args.plot_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
update_inhibation_weights = args.update_inhibation_weights
sim_params = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
              'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
              'stuck_at_fault': args.stuck_at_fault, 'retention_loss': args.retention_loss,
              'aging_effect': args.aging_effect, 'wire_width': args.wire_width, 'input_bit': args.input_bit,
              'batch_interval': 1, 'CMOS_technode': args.CMOS_technode, 'ADC_precision': args.ADC_precision,
              'ADC_setting': args.ADC_setting,'ADC_rounding_function': args.ADC_rounding_function,
              'device_roadmap': args.device_roadmap, 'temperature': args.temperature,
              'hardware_estimation': args.hardware_estimation}

# %% Sets up Gpu use
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(seed)
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = torch.cuda.is_available() * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# %% Multiple test
out_root = 'Test_Accuracy_Results.txt'
multiple_test_no = 10

for test_cnt in range(multiple_test_no):
    out = open(out_root, 'a')
    # %% Build network.
    network = IncreasingInhibitionNetwork(
        n_input=784,
        n_neurons=n_neurons,
        start_inhib=10,
        max_inhib=-40,
        theta_plus=0.05,
        tc_theta_decay=1e7,
        inpt_shape=(1, 28, 28),
        nu=(1e-4, 1e-2),
        sim_params=sim_params,
        batch_size=train_batch_size
    )
    
    network.to(device)

    # Area print
    if sim_params['device_name'] != 'trace' and sim_params['hardware_estimation']:
        total_area = 0
        for l in network.layers:
            network.layers[l].transform.total_area_calculation()
            total_area += network.layers[l].transform.sim_area['sim_total_area']
        print("total crossbar area=", total_area, " m2")
    
    # %% Load MNIST data.
    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
    
    # Record spikes during the simulation.
    spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)
    
    # Neuron assignments and spike proportions.
    n_classes = 10
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)
    
    # Sequence of accuracy estimates.
    accuracy = {"all": [], "proportion": []}
    
    # Voltage recording for excitatory and inhibitory layers.
    som_voltage_monitor = Monitor(
        network.layers["Y"], ["v"], time=int(time / dt), device=device
    )
    network.add_monitor(som_voltage_monitor, name="som_voltage")
    
    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)
        
    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
        )
        network.add_monitor(voltages[layer], name="%s_voltages" % layer)
    
    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None
    save_weights_fn = "plots/weights/weights.png"
    save_performance_fn = "plots/performance/performance.png"
    save_assignments_fn = "plots/assignments/assignments.png"
    
    directorys = ["plots", "plots/weights", "plots/performance", "plots/assignments"]
    for directory in directorys:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # diagonal weights for increasing the inhibition
    weights_mask = (1 - torch.diag(torch.ones(n_neurons))).to(device)
    
    # %% Train the network.
    print("\nBegin training.\n")
    start = t()
    
    for epoch in range(n_epochs):
        labels = []
    
        if epoch % progress_interval == 0:
            print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
            start = t()
    
        # Create a dataloader to iterate and batch data
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=n_workers, 
            pin_memory=gpu
        )
        
        pbar = tqdm(total=n_train)
        for step, batch in enumerate(dataloader):
            if step == n_train:
                break
    
            # Get next input sample.
            inputs = {"X": batch["encoded_image"].transpose(0, 1).to(device)} 
    
            if step > 0:
                if (step * train_batch_size) % update_inhibation_weights == 0:#update_inhibation_weights=500
                    if (step*train_batch_size) % (update_inhibation_weights * 10) == 0:
                        network.Y_to_Y.w -= weights_mask * 50
                    else:
                        # Inhibit the connection even more
                        network.Y_to_Y.w -= weights_mask * 0.5
    
                if (step*train_batch_size) % update_interval == 0:
                    # Convert the array of labels into a tensor
                    label_tensor = torch.tensor(labels, device=device)

                    # Assign labels to excitatory layer neurons.
                    assignments, proportions, rates = assign_labels(
                        spikes=spike_record,
                        labels=label_tensor,
                        n_labels=n_classes,
                        rates=rates,
                    )
    
                    labels = []
    
            # labels.append(batch["label"])
            labels.extend(batch["label"].tolist())
            
            # Run the network on the input.
            temp_spikes = 0
            network.run(inputs=inputs, time=time, input_time_dim=1)
            temp_spikes = spikes["Y"].get("s").permute((1, 0, 2))

            # Get voltage recording.
            exc_voltages = som_voltage_monitor.get("v")
    
            # Add to spikes recording.
            spike_record[
                (step * train_batch_size) % update_interval : (step * train_batch_size) % update_interval + temp_spikes.size(0)
                ].copy_(temp_spikes, non_blocking=True)
    
            # %% Update
            network.reset_state_variables()  # Reset state variables.
            if epoch < n_epochs - 1 or step < n_train - 1:
                network.mem_t_update()
            pbar.set_description_str("Train progress: ")
            pbar.update()
    
    print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
    print("Training complete.\n")


    # %% Load MNIST data.
    test_dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
    
    # Create a dataloader to iterate and batch data
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_workers, pin_memory=gpu
    )
    
    # Sequence of accuracy estimates.
    accuracy = {"all": 0, "proportion": 0}
    
    # %% Test the network.
    print("\nBegin testing\n")
    network.train(mode=False)
    start = t()

    print("Test progress: ")
    for batch in tqdm(test_dataloader):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].transpose(0, 1).to(device)}
    
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
    
        spike_record = spikes["Y"].get("s").transpose(0, 1)

        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(batch["label"], device=device)
    
        # Get network predictions.
        all_activity_pred = all_activity(
            spikes=spike_record, assignments=assignments, n_labels=n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )
    
        # Compute network accuracy according to available classification strategies.
        accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
        accuracy["proportion"] += float(
            torch.sum(label_tensor.long() == proportion_pred).item()
        )
    
        network.reset_state_variables()  # Reset state variables.
    

    print("\nAll activity accuracy: %.4f" % (accuracy["all"] / args.n_test))
    print("Proportion weighting accuracy: %.4f \n" % (accuracy["proportion"] / args.n_test))

    print("Testing complete.\n")
    
    # %% output+clear
    out_txt = 'All activity accuracy:' + str(accuracy["all"] / args.n_test) +\
                '\tProportion weighting accuracy:' + str(accuracy["proportion"] / args.n_test) + '\n'
    out.write(out_txt)
    out.close()
    
    del network

