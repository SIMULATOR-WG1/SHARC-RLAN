# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:45:31 2019

@author: Jussif
"""

import unittest

from sharc.antenna.antenna_laa_r5a976 import AntennaElementLaar5a
from sharc.support.named_tuples import AntennaPar

import numpy as np
import numpy.testing as npt
#import sys

class AntennaElementLaar5a_Test(unittest.TestCase):

    def setUp(self):

        norm = False
        norm_file = None
        element_pattern = "FIXED"
        element_max_g = 5
        element_phi_deg_3db = 3.5
        element_theta_deg_3db = 40
        element_am = 30
        element_sla_v = 20
        n_rows = 8
        n_columns = 8
        horiz_spacing = 0.5
        vert_spacing = 0.5
        down_tilt = -10

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
        
        self.antenna = AntennaElementLaar5a(par)
        

    def test_calculate_gain(self):        
        phi_vec = np.array([-180,-140,-100,-60,-20,20,60,100,140])
    
        theta_vec = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85])
        
        pattern_hor_0deg = np.zeros(phi_vec.shape)
        pattern_ver_0deg = np.zeros(theta_vec.shape)

        ref_gain_h = np.array([2,2.66,3.97,4.82,4.99,4.99,4.81,3.97,2.66])
        ref_gain_v = np.array([5,4.81,4.25,3.31,2,0.31,-1.75,-4.19,-7,-10.19,-13.75,-15,-15,-15,-15,-15,-15,-15,])
        
        for phi, index in zip(phi_vec, range(len(phi_vec))):            
            pattern_hor_0deg[index] = self.antenna.element_pattern(phi,0)
        
        for theta, index in zip(theta_vec, range(len(theta_vec))):            
            pattern_ver_0deg[index] = self.antenna.element_pattern(0,theta)
            
        npt.assert_allclose(pattern_hor_0deg, ref_gain_h, atol=1e-2)
        
        npt.assert_allclose(pattern_ver_0deg, ref_gain_v, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
