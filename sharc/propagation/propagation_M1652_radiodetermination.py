# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:04:27 2017

@author: edgar
"""

from sharc.propagation.propagation import Propagation

import numpy as np
import random

class PropagationM1652(Propagation):
    """
    Implements the Free Space propagation model.
    Frequency in MHz and distance in meters
    """

    def get_loss(self, *args, **kwargs) -> np.array:
        if "distance_2D" in kwargs:
            d = kwargs["distance_2D"]
        else:
            d = kwargs["distance_3D"]

        
        number_of_sectors = kwargs.pop("number_of_sectors",1)

        loss = 35*np.log10(d) + random.randint(0,20)

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss
