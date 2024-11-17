#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:49:32 2022

@author: zhangxin
"""

import os
import sys
import torch
import argparse
import numpy as np
import math

from torchvision import transforms
from tqdm import tqdm

from time import time as t

sys.path.append('../../')

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder, poisson
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

# %% Argument
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--train_batch_size", type=int, default=50)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--multiple_test_no", type=int, default=90)
parser.add_argument("--n_epochs", type=int, default=3)
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
parser.add_argument("--memristor_device", type=str, default='ferro')  # trace: original trace
parser.add_argument("--c2c_variation", type=bool, default=True)
parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
multiple_test_no = args.multiple_test_no
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
device_params = {'device_name': args.memristor_device, 'c2c_variation': args.c2c_variation}

# %% Sets up Gpu use
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))

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
out_root = 'Accuracy_Results_fast_train.txt'

# %% Load MNIST training data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# %% Load MNIST test data.
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

for test_cnt in range(multiple_test_no):
    out = open(out_root, 'a')

    # %% Enable test while training
    signal_break = 0
    tmp_acc = 0
    best_acc = 0
    best_capacity = 0
    patience = 10

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
        mem_device=device_params,
        batch_size=train_batch_size
    )

    network.to(device)

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

            network.train(mode=True)

            # Get next input sample.
            inputs = {"X": batch["encoded_image"].transpose(0, 1).to(device)}

            if step > 0:
                if (step * train_batch_size) % update_inhibation_weights == 0:  # update_inhibation_weights=500
                    if (step * train_batch_size) % (update_inhibation_weights * 10) == 0:
                        network.Y_to_Y.w -= weights_mask * 50
                    else:
                        # Inhibit the connection even more
                        network.Y_to_Y.w -= weights_mask * 0.5

                if (step * train_batch_size) % update_interval == 0:
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
            (step * train_batch_size) % update_interval: (step * train_batch_size) % update_interval + temp_spikes.size(
                0)
            ].copy_(temp_spikes, non_blocking=True)

            # %% Update
            network.reset_state_variables()  # Reset state variables.
            pbar.set_description_str("Train progress: ")
            pbar.update()

            # %% Test while training
            network.train(mode=False)

            if (step >= 500 or epoch > 0) and ((step * train_batch_size) % (update_interval * 20) == 0):
                accuracy_test = {"all": 0, "proportion": 0}
                print("\nBegin testing while training\n")
                for batch_test in tqdm(test_dataloader):
                    # Get next input sample.
                    inputs_test = {"X": batch_test["encoded_image"].transpose(0, 1).to(device)}

                    # Run the network on the input.
                    network.run(inputs=inputs_test, time=time, input_time_dim=1)

                    spike_record_test = spikes["Y"].get("s").transpose(0, 1)
                    label_tensor_test = torch.tensor(batch_test["label"], device=device)

                    # Get network predictions.
                    all_activity_pred_test = all_activity(
                        spikes=spike_record_test, assignments=assignments, n_labels=n_classes
                    )
                    proportion_pred_test = proportion_weighting(
                        spikes=spike_record_test,
                        assignments=assignments,
                        proportions=proportions,
                        n_labels=n_classes,
                    )

                    # Compute network accuracy according to available classification strategies.
                    accuracy_test["all"] += float(torch.sum(label_tensor_test.long() == all_activity_pred_test).item())
                    accuracy_test["proportion"] += float(
                        torch.sum(label_tensor_test.long() == proportion_pred_test).item()
                    )
                    network.reset_state_variables()

                # Strategies of when to end the training
                tmp_acc = accuracy_test["all"] * 100 / args.n_test
                if tmp_acc >= best_acc:
                    best_acc = tmp_acc
                    signal_break = 0
                    best_capacity = epoch * args.n_train + (step + 1) * train_batch_size
                else:
                    signal_break += 1

                total_capacity = epoch * args.n_train + (step + 1) * train_batch_size

                print("\nCurrent all activity accuracy: %.4f" % (accuracy_test["all"] / args.n_test))
                print("\nBest all activity accuracy: %.4f, Best capacity: %d, Signal break:%d.\n" % \
                      (best_acc, best_capacity, signal_break))
                print("Testing while training complete.\n")

                if signal_break >= patience:
                    break

        if signal_break >= patience:
            break

    print("Best_acc: %.4f" % best_acc)

    # %% output+clear
    out_txt = 'All activity accuracy:' + str(best_acc) + '\tbest capacity:' + str(
        best_capacity) + '\ttest capacity:' + str(total_capacity) + '\n'
    out.write(out_txt)
    out.close()

    del network

