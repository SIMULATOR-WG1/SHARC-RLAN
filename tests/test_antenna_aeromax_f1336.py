# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:01:30 2019

@author: Jussif
"""

import unittest

from sharc.antenna.antenna_element_aeromax_f1336 import AntennaElementAeromaxF1336
from sharc.support.named_tuples import AntennaPar

import numpy as np
import numpy.testing as npt
#import sys

class AntennaElementAeromaxF1336_Test(unittest.TestCase):

    def setUp(self):

        norm = False
        norm_file = None
        element_pattern = "FIXED"
        element_max_g = 0
        element_phi_deg_3db = 360
        element_theta_deg_3db = 90
        element_am = 30
        element_sla_v = 30
        n_rows = 8
        n_columns = 8
        horiz_spacing = 0.5
        vert_spacing = 0.5
        down_tilt = 0

        par = AntennaPar(norm,
                         norm_file,
                         element_pattern,
                         element_max_g,
                         element_phi_deg_3db,
                         element_theta_deg_3db,
                         element_am,
                         element_sla_v,
                         n_rows,
                         n_columns,
                         horiz_spacing,
                         vert_spacing,
                         down_tilt)
        
        self.antenna = AntennaElementAeromaxF1336(par)
        

    def test_calculate_gain(self):        
        phi_vec = np.array([-180,-140,-100,-60,-20,20,60,100,140,180])
    
        theta_vec = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
        
        pattern_hor_0deg = np.zeros(phi_vec.shape)
        pattern_ver_0deg = np.zeros(theta_vec.shape)

        ref_gain_h = np.array([-3,-1.81,-0.92,-0.33,-0.04,-0.04,-0.33,-0.92,-1.81,-3])
        ref_gain_v = np.array([-0,-0.84,-5.06,-7.35,-8.84,-9.90,-10.68,-11.2902,
                               -11.7692,-12.1558,-12.473,-12.7368,-12.959,-13.148,
                               -13.31,-13.45,-13.57,-13.68,-13.78])
        
        for phi, index in zip(phi_vec, range(len(phi_vec))):            
            pattern_hor_0deg[index] = self.antenna.element_pattern(phi,0)
        
        for theta, index in zip(theta_vec, range(len(theta_vec))):            
            pattern_ver_0deg[index] = self.antenna.element_pattern(0,theta)
            
        npt.assert_allclose(pattern_hor_0deg, ref_gain_h, atol=1e-2)
        
        npt.assert_allclose(pattern_ver_0deg, ref_gain_v, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
