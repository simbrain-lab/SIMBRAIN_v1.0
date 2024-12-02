import torch
from typing import Iterable, Optional, Union
import json

class Area(torch.nn.Module):
    # language=rst
    """
    Abstract base class for area estimation of memristor crossbar.
    """
    
    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        length_row: float = 0.0,
        length_col: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param shape: The dimensionality of the layer.
        :param length_row: The physical length of the horizontal wire in the crossbar.
        :param length_col: The physical length of the vertical wire in the crossbar.
        :param array_area:  The area of memristor array.
        """
        super().__init__()

        self.shape = shape
        self.length_row = length_row
        self.length_col = length_col

        self.array_area = self.cal_area()


    def cal_area(self):
        # language=rst
        """
        Calculate the area of memristor array.
        """
        array_area_value = self.length_row * self.length_col
        return array_area_value