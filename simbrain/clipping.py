from typing import Iterable, Optional, Union
import torch
import json


class Clipping(torch.nn.Module):
    # language=rst
    """
    Abstract base class for DAC module.
    """

    def __init__(
            self,
            sim_params: dict = {},
            shape: Optional[Iterable[int]] = None,
            memristor_info_dict: dict = {},
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        """
        super().__init__()

        self.shape = shape
        self.sim_params = sim_params
        self.memristor_info_dict = memristor_info_dict
        self.device_name = sim_params['device_name']
        self.Gon = self.memristor_info_dict[self.device_name]['G_on']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.read_v_amp = self.memristor_info_dict[self.device_name]['v_read']
        self.clipping = sim_params['clipping']
        if self.clipping:
            self.clipping_threshold = sim_params['clipping_threshold']
            self.clipping_threshold = self.read_v_amp * (
                        self.clipping_threshold * self.Goff + (1 - self.clipping_threshold) * self.Gon)

    def clipping_function(self, mem_i_origin) -> None:
        if self.clipping:
            select_signal = self.comparator(mem_i_origin)
            mem_i = self.mux(select_signal, mem_i_origin)
        else:
            mem_i = mem_i_origin
        return mem_i

    def comparator(self, input_signal):
        return input_signal >= self.clipping_threshold

    def mux(self, select_signal, input_signal):
        # if select_signal is True, output input_signal, else output 0
        return torch.where(select_signal == True, input_signal, self.read_v_amp * self.Gon)