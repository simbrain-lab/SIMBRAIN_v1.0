[TOC]

# Introduction

This project develops a SImulation framework for Memristor-Based Realistic Architectures with nonIdealities in Neuro-inspired learning (SIMBRAIN). SIMBRAIN offers a variety of flexible design options to estimate both NN accuracy and circuit-level performance to facilitate design space exploration.

# Key Features

SIMBRAIN integrates a full fitting flow for modeling characteristics and nonidealities of the memristor to translate the raw device data into an integrated versatile behavioral model that accounts for all three types of nonidealities.

SIMBRAIN focuses on passive analog memristor array architectures for their potential in achieving high integration density and high-speed operations, which are promising in the field of high-performance computing.

SIMBRAIN proposes a reconfigurable crossbar design, which can not only map trace-based SNN to crossbar-based architecture for unsupervised STDP online learning, but is also compatible with ANNs, including inference and in-situ training of multilayer perceptron (MLP) and convolutional neural network (CNN).

# Installation Requirements

### Clone the Repository

```bash
git clone https://github.com/simbrain-lab/SIMBRAIN_v1.0.git
```

### Basic Requirements

Required platform:	`Python 3.8`,  `PyTorch 1.13`

Required packages:	`numpy`,  `pandas`,  `matplotlib`

### Requirements for different scenarios

Scenario 1: Device to Model:	`scikit-learn`,  `psutil`

Scenario 2: Model to NN:	`torchvision`,  `opencv-python`,  `tqdm`

Scenario 3: Device to NN:	Both the required packages for scenario 1 and scenario 2.

# Usage（Quick Start）

SIMBRAIN provides some examples for three scenarios.

### Scenario 1: Device to Model

Some examples of device-to-model simulation for characteristics and nonidealities of the memristor are stored in`examples/Memristor_Modeling`.  `Baseline_model`, `Variation`, `Aging_effect`,  `Retention loss` and `SAF` providing examples to simulate separate characteristic or nonideality of the memristor. To take all nonidealities as considered, use `full_fitting_flow.py`  as follows:

```python
from Memristor_Modeling.full_fitting_flow import full_fitting

sim_params = full_fitting(None, None)
```

### Scenario 2: Model to NN

Some examples of model-to-NN simulation are stored in `examples/MLP`,`examples/CNN` and `examples/SNN`.

- `Model_to_MLP_inference.py`, `Model_to_MLP_inference_retention.py` and `Model_to_MLP_training.py` for MLP.
- `Model_to_CNN_inference.py` for CNN.
- `Model_to_STDP_dynamic_train.py` and `Model_to_MLP_earlystop.py` for SNN.

### Scenario 3: Device to NN

Some examples of device-to-NN simulation are stored in `examples/MLP`,`examples/CNN` and `examples/SNN`.

- `Device_to_MLP_inference.py` and `Device_to_MLP_training.py` for MLP.
- `Device_to_CNN_inference.py` for CNN.
- `Device_to_STDP_dynamic_train.py` for SNN.

### Simulation Parameters

The simulation parameters include the parameters for the memristor crossbar and the parameters for the peripheral circuit used in model-to-NN simulation or device-to-NN simulation.

Memristor device configuration(only in model-to-NN simulation)：

|     Parameter      |                          Definition                          | Default |
| :----------------: | :----------------------------------------------------------: | :-----: |
| --memristor_device | Choose the memristor model, five are provided here: 'ideal', 'ferro', 'MF', 'CMS', and 'mine' | 'ideal' |
|  --c2c_variation   | Whether to include the cycle-to-cycle variation in the simulation: True, False |  False  |
|  --d2d_variation   | Whether to include the device-to-device variation in the simulation: 0, 1, 2, 3. 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only |    0    |
|  --stuck_at_fault  | Whether to include the stuck at fault in the simulation: True, False |  False  |
|  --retention_loss  | Whether to include the retention loss in the simulation: True, False |  False  |
|   --aging_effect   | Whether to include the aging effect in the simulation: 0, 1, 2. 0: No aging effect, 1: equation 1, 2: equation 2 |    0    |

Circuit configuration：

|        Parameter        |                          Definition                          |  Default   |
| :---------------------: | :----------------------------------------------------------: | :--------: |
|  --memristor_structure  | Chose the memristor structure, three are provided here: 'trace', 'crossbar', and 'STDP_crossbar' | 'crossbar' |
|       --input_bit       |   The DAC resolution: int (1 for STDP, 2-64 for MLP & CNN)   |     8      |
|     --ADC_precision     |                The ADC resolution: int (2-32)                |     16     |
|      --ADC_setting      | 2 or 4. Employing four sets equips each memristor crossbar with a dedicated ADC. Using two sets integrates two crossbars vertically due to their summable currents per column, allowing them to share a single ADC set. |     4      |
| --ADC_rounding_function |                      'floor' or 'round'                      |  'floor'   |
|      --wire_width       | In practice, the wire width shall be set around 1/2 of the memristor size: ideal - 200 (200 um), ferro - 200 (200nm), MF - 10000(10um), CMS - 10000(10um) |   10000    |
|     --CMOS_technode     | Technology node for the peripheral circuits: 130, 90, 65, 45, 32, 22, 14, 10, 7 (nm) |     45     |
|    --device_roadmap     |          High performance or low power: 'HP', 'LP'           |    'HP'    |
|      --temperature      |                      Default to 300 (K)                      |    300     |
|  --hardware_estimation  |       Whether to run hardware estimation: True, False        |    True    |



# Directory Structure

- ├── **simbrain**                                                  // Main project directory

- │   ├── `memristor_fit.py`                           // Fit memristor models

- │   ├── **Fitting_Functions**                             // Store memristor models

- │   ├── `formula.py`                                       // Calculate layout area, resistance, capacitance

- │   ├── `mapping.py`                                       // Map NNs to the memristor array

- │   ├── `memarray.py`                                     // The MemArray module

- │   ├── `memarea.py`                                       // Calculate area of the memristor array

- │   ├── `mempower.py`                                     // Calculate power of the memristor array

- │   ├── `periphcircuit.py`                           // The Periphery module

- │   ├── `peripharea.py`                                 // Calculate area of the periphery circuits

- │   └── `periphcipower.py`                           // Calculate power of the periphery circuits

- ├── **memristor_data**                                    // Raw memristor data and configuration for use

- ├── **reference_memristor_data**                // Store raw memristor data and configuration

- │   ├── `my_memristor.json`                         // Store parameters of memristor models

- │   ├── `sim_params.json`                             // Store fitting configurations

- │   └── `*.xlsx`                                          // Store raw memristor data

- ├── **examples**                                                 // Store examples for different simulations

- │   ├── **Memristor_Modeling**                      // Store examples for device-to-model simulation

- │   ├── **MLP**                                                      // Store examples for device/model-to-MLP simulation

- │   ├── **CNN**                                                     // Store examples for device/model-to-CNN simulation

- │   └── **SNN**                                                     // Store examples for device/model-to-SNN simulation

- ├── `CMOS_tech_info.json`                          // Store technology information for CMOS devices in peripheral circuits

- ├── `wire_tech_info.json`                          // Store technology information for wires

- ├── `memristor_device_info.json`            // Store memristor parameters for NN simulation

- └── `memristor_lut.pkl`                               // Store memristor parameters and W2V LUT for ANN simulation

# Updates

Version 1.0

# Contributing

...