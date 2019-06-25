# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:53:00 2019

@author: Jussif
"""

import unittest

from sharc.antenna.antenna_r1652_1 import AntennaRadar
from sharc.parameters.parameters_fss_es import ParametersFssEs

import numpy as np
import numpy.testing as npt

class AntennaRadarTest(unittest.TestCase):

    def setUp(self):
        param = ParametersFssEs()
        param.antenna_pattern = "ITU-R M.1652-1"
        param.frequency = 5.100
        param.antenna_gain = 40
        param.diameter = 5
        self.antenna = AntennaRadar(param)


    def test_calculate_gain(self):
        tetha = np.array([0, 9,18,27,36,45,54,63,72,81,90,99,108,117,126,135,144,153,162,171,180])

        ref_gain = np.array([40,9.15,1.62,-2.78,-5.91,-8.33,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9])
        gain = self.antenna.calculate_gain(off_axis_angle_vec=tetha) 
        npt.assert_allclose(gain, ref_gain, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
