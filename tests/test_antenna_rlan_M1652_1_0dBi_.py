# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:02:03 2019

@author: Jussif
"""

import unittest

from sharc.antenna.antenna_rlan_M1652_1_0dBi_ import AntennaElementM1652_1_0dbi
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
        
        self.antenna = AntennaElementM1652_1_0dbi(par)
        

    def test_calculate_gain(self):        
        phi_vec = np.array([-180,-140,-100,-60,-20,20,60,100,140])
    
        theta_vec = np.array([-90,-70,-50,-30,-10,10,30,50,70,90])
        
        pattern_hor_0deg = np.zeros(phi_vec.shape)
        pattern_ver_0deg = np.zeros(theta_vec.shape)

        ref_gain_h = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,])
        ref_gain_v = np.array([-5,-5,-6,-6,-1,0,0,-4,-4,-4])
        
        for phi, index in zip(phi_vec, range(len(phi_vec))):            
            pattern_hor_0deg[index] = self.antenna.element_pattern(phi,0)
        
        for theta, index in zip(theta_vec, range(len(theta_vec))):            
            pattern_ver_0deg[index] = self.antenna.element_pattern(0,theta)
            
        npt.assert_allclose(pattern_hor_0deg, ref_gain_h, atol=1e-2)
        
        npt.assert_allclose(pattern_ver_0deg, ref_gain_v, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
