from typing import Iterable, Optional, Union
from simbrain.periphpower import DAC_Module_Power
from simbrain.periphpower import ADC_Module_Power
from simbrain.peripharea import DAC_Module_Area
from simbrain.peripharea import ADC_Module_Area
import torch
import json

class DAC_Module(torch.nn.Module):
    # language=rst
    """
    Abstract base class for DAC module.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        :param CMOS_tech_info_dict: The parameters of CMOS technology.
        """
        super().__init__()

        self.shape = shape
        self.sim_params = sim_params
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.memristor_info_dict = memristor_info_dict
        self.device_structure = sim_params['device_structure']
        self.input_bit = sim_params['input_bit']
        self.device_name = sim_params['device_name']
        self.read_v_amp = self.memristor_info_dict[self.device_name]['v_read']

        if self.sim_params['hardware_estimation']:
            self.DAC_module_power = DAC_Module_Power(sim_params=self.sim_params, shape=self.shape,
                                                     CMOS_tech_info_dict=self.CMOS_tech_info_dict,
                                                     memristor_info_dict=self.memristor_info_dict)
            self.DAC_module_area = DAC_Module_Area(sim_params=sim_params, shape=self.shape,
                                                   CMOS_tech_info_dict=self.CMOS_tech_info_dict,
                                                   memristor_info_dict=self.memristor_info_dict)


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        if self.sim_params['hardware_estimation']:
            self.DAC_module_power.set_batch_size(batch_size=self.batch_size)


    def DAC_read(self, mem_v, sgn) -> None:
        # language=rst
        """
        Performs the DAC operation during memristor read operation, also estimates the energy consumption for the hardware during the read process if hardware estimation is enabled.
    
        :param mem_v: Voltage inputs to the memristor array during the read operation.
        :param sgn: A string indicating the sign of the read operation. Can be 'pos' for positive or 'neg' for negative, which determines how the voltage values are processed.
        """
        if self.device_structure in {'trace', 'STDP_crossbar'}:
            activity_read = torch.nonzero(mem_v).size(0) / mem_v.numel()
            if self.sim_params['hardware_estimation']:
                self.DAC_module_power.switch_matrix_read_energy_calculation(activity_read=activity_read, mem_v_shape=mem_v.shape)
            return mem_v

        else:
            # mem_v shape [write_batch_size, read_batch_size, row_no]
            # increase one dimension of the input by input_bit
            read_sequence = torch.zeros(self.input_bit, *(mem_v.shape), device=mem_v.device, dtype=bool)

            # positive read sequence generation
            if sgn == 'pos':
                v_read = torch.relu(mem_v)
            elif sgn == 'neg':
                v_read = torch.relu(mem_v * -1)

            mem_v = None
            v_read = torch.round(v_read * (2 ** self.input_bit - 1))
            v_read = torch.clamp(v_read, 0, 2 ** self.input_bit - 1)

            if self.input_bit <= 8:
                v_read = v_read.to(torch.uint8)
            else:
                v_read = v_read.to(torch.int64)
            # TODO 16 32 64
            for i in range(self.input_bit):
                bit = torch.bitwise_and(v_read, 2 ** i).bool()
                read_sequence[i] = bit
            v_read = None
            bit = None

            activity_read = read_sequence.sum().item() / read_sequence.numel()
            if self.sim_params['hardware_estimation']:
                self.DAC_module_power.switch_matrix_read_energy_calculation(activity_read=activity_read, mem_v_shape=read_sequence.shape)

            return read_sequence

    def DAC_write(self, mem_v, mem_v_amp) -> None:
        # language=rst
        """
        Performs power calculations of DAC during memristor write operation for different hardware estimations based on the device structure.
    
        :param mem_v: Voltage inputs to the memristor array during the write operation.
        :param mem_v_amp: Amplified voltage inputs to the memristor array during the write operation.
        """
        if self.sim_params['hardware_estimation']:
            if self.device_structure == 'trace':
                self.DAC_module_power.switch_matrix_col_write_energy_calculation(mem_v=mem_v)
            elif self.device_structure == 'crossbar':
                self.DAC_module_power.switch_matrix_row_write_energy_calculation(mem_v_amp=mem_v_amp)
                self.DAC_module_power.switch_matrix_col_write_energy_calculation(mem_v=mem_v)
            elif self.device_structure == 'STDP_crossbar':
                self.DAC_module_power.switch_matrix_row_write_energy_calculation(mem_v_amp=mem_v_amp[0])
                mem_v_pos = torch.where(mem_v>0, 0, mem_v_amp[0])
                self.DAC_module_power.switch_matrix_col_write_energy_calculation(mem_v=mem_v_pos)
                self.DAC_module_power.switch_matrix_row_write_energy_calculation(mem_v_amp=mem_v_amp[1])
                mem_v_neg = torch.where(mem_v<0, 0, mem_v_amp[1])
                self.DAC_module_power.switch_matrix_col_write_energy_calculation(mem_v=mem_v_neg)
            else:
                raise Exception("Only trace, mimo and crossbar architecture are supported!")


    def DAC_reset(self, mem_v) -> None:
        # language=rst
        """
        Performs power calculations of DAC during memristor reset operation.
    
        :param mem_v: Voltage inputs to the memristor array during the reset operation.
        """
        if self.sim_params['hardware_estimation']:
            self.DAC_module_power.switch_matrix_reset_energy_calculation(mem_v=mem_v)

    def DAC_energy_calculation(self, mem_t) -> None:
        # language=rst
        """
        Calculates the total energy consumption of the DAC.

        :param mem_t: Time of the memristor crossbar.
        """
        self.DAC_module_power.DAC_energy_calculation(mem_t=mem_t)


class ADC_Module(torch.nn.Module):
    # language=rst
    """
    Abstract base class for ADC and ShiftAdd module.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        :param CMOS_tech_info_dict: The parameters of CMOS technology.
        """
        super().__init__()

        self.shape = shape
        self.sim_params = sim_params
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.memristor_info_dict = memristor_info_dict
        self.input_bit = sim_params['input_bit']
        self.ADC_precision = sim_params['ADC_precision']
        self.ADC_rounding_function = sim_params['ADC_rounding_function']
        self.device_name = sim_params['device_name']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.read_v_amp = self.memristor_info_dict[self.device_name]['v_read']
        self.device_structure = sim_params['device_structure']

        if self.sim_params['hardware_estimation']:
            if self.ADC_rounding_function == 'floor':
                self.ADC_module_power = ADC_Module_Power(sim_params=self.sim_params,
                                                         shape=self.shape,
                                                         CMOS_tech_info_dict=self.CMOS_tech_info_dict,
                                                         memristor_info_dict=self.memristor_info_dict)
                self.ADC_module_area = ADC_Module_Area(sim_params=sim_params,
                                                       shape=self.shape,
                                                       CMOS_tech_info_dict=self.CMOS_tech_info_dict,
                                                       memristor_info_dict=self.memristor_info_dict)
            else:
                raise Exception("Only floor function ADC supports hardware estimation!")


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        if self.sim_params['hardware_estimation']:
            self.ADC_module_power.set_batch_size(batch_size=self.batch_size)


    def ADC_read(self, mem_i_sequence, mem_c, high_cut_ratio) -> None:
        # language=rst
        """
        Performs the ADC operation during memristor read operation, quantizing the input current sequence and calculating the output current.

        :param mem_i_sequence: The sequence of input currents of ADC to be quantized.
        :param mem_c: Conductance of memristor array.
        :param high_cut_ratio: The ratio used to determine the maximum expected current during the read operation.
        """
        # Initial mem_i
        mem_i = torch.zeros(mem_i_sequence.size()[1:], device=mem_i_sequence.device)

        # calculate the theoretical max and min
        mem_i_max = high_cut_ratio * torch.sum(self.read_v_amp*mem_c, dim=0)
        mem_i_min = 0
        mem_i_step = (mem_i_max - mem_i_min) / (2**self.ADC_precision)
        mem_i_index = (mem_i_sequence - mem_i_min) / mem_i_step
        if self.ADC_rounding_function == 'round':
            mem_i_index = torch.clamp(torch.floor(mem_i_index + 0.5), 0, 2**self.ADC_precision-1)
        elif self.ADC_rounding_function == 'floor':
            mem_i_index = torch.clamp(torch.floor(mem_i_index), 0, 2**self.ADC_precision-1)
        else:
            raise Exception("Only round and floor function are supported!")
        mem_i_sequence_quantized = mem_i_index * mem_i_step + mem_i_min

        # Shift add to get the output current
        for i in range(self.input_bit):
            mem_i += mem_i_sequence_quantized[i, :, :, :] * 2 ** i

        if self.sim_params['hardware_estimation']:
            self.ADC_module_power.SarADC_energy_calculation(mem_i_sequence=mem_i_sequence)
            if self.device_structure == 'crossbar':
                self.ADC_module_power.shift_add_energy_calculation(mem_i_sequence=mem_i_sequence)

        return mem_i

    def ADC_energy_calculation(self, mem_t) -> None:
        # language=rst
        """
        Calculates the total energy consumption of the ADC.

        :param mem_t: Time of the memristor crossbar.
        """
        self.ADC_module_power.ADC_energy_calculation(mem_t=mem_t)
