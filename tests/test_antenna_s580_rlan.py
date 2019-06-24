# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:24:14 2019

@author: Jussif
"""

import unittest

from sharc.antenna.antenna_s580_rlan import AntennaS580_rlan
from sharc.parameters.parameters_fss_es import ParametersFssEs

import numpy as np
import numpy.testing as npt

class AntennaS580_rlanTest(unittest.TestCase):

    def setUp(self):
        param = ParametersFssEs()
        param.antenna_pattern = "ITU-R S.580-6"
        param.frequency = 5.100
        param.antenna_gain = 35
        param.diameter = 5
        self.antenna = AntennaS580_rlan(param)
        

    def test_calculate_gain(self):
        psi = np.array([2, 10.9, 19.8, 28.7, 37.6, 46.5, 55.4, 64.3, 73.2,  
                        82.1,91, 99.9, 108.8,117.7,126.6,135.5,144.4,153.3,162.2,171.1,180])
        ref_gain = np.array([35,35,35,-4.44705,-7.3797,-9.68632,-10,-10,-10,-10,-10, 
                             -10,-10,-10,-10,-10,-10,-10,-10,-10,-10])
        gain = self.antenna.calculate_gain(off_axis_angle_vec=psi) #- self.antenna.peak_gain
        npt.assert_allclose(gain, ref_gain, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
