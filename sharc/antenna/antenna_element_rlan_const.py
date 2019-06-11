# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:13:58 2017

@author: Calil
"""


from sharc.support.named_tuples import AntennaPar
import numpy as np

class AntennaElementRlanConst(object):
    """
    Implements a single element of an RLAN antenna array with constant gain

    Attributes
    ----------
        g_max (float): maximum gain of element

    """

    def __init__(self,par: AntennaPar):
        """
        Constructs an AntennaElementRlan object.

        Parameters
        ---------
            param (ParametersAntennaRlan): antenna RLAN parameters
        """
        self.g_max = par.element_max_g

    def element_pattern(self, phi: np.array, theta: np.array) -> np.array:
        """
        Calculates the element radiation pattern gain.

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]
            phi (np.array): azimuth angle [degrees]

        Returns
        -------
            gain (np.array): element radiation pattern gain value
        """
        gain = np.ones(np.shape(phi)) * self.g_max

        return gain
