# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:32:30 2017

@author: edgar
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.simulation_uplink import SimulationUplink
from sharc.parameters.parameters_imt import ParametersImt
from sharc.parameters.parameters_fss import ParametersFss
from sharc.parameters.parameters_antenna_imt import ParametersAntennaImt
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.station_factory import StationFactory

class SimulationUplinkTest(unittest.TestCase):
    
    def setUp(self):
        self.param = ParametersImt()
        self.param.topology = "SINGLE_BS"
        self.param.num_macrocell_sites = 19
        self.param.num_clusters = 2
        self.param.intersite_distance = 200
        self.param.minimum_separation_distance_bs_ue = 10
        self.param.interfered_with = False
        self.param.frequency = 10000
        self.param.bandwidth = 100
        self.param.rb_bandwidth = 0.180
        self.param.guard_band_ratio = 0.1
        self.param.ho_margin = 3
        self.param.bs_load_probability = 1
        self.param.num_resource_blocks = 10
        self.param.bs_tx_power = 40
        self.param.bs_height = 6
        self.param.bs_aclr = 40
        self.param.bs_acs = 30
        self.param.bs_noise_figure = 7
        self.param.bs_noise_temperature = 290
        self.param.bs_feed_loss = 3
        self.param.ul_attenuation_factor = 0.4
        self.param.ul_sinr_min = -10
        self.param.ul_sinr_max = 22
        self.param.ue_k = 2
        self.param.ue_k_m = 1
        self.param.ue_tx_power_control = "OFF"
        self.param.ue_tx_power_target = -95
        self.param.ue_tx_power_alfa = 0.8
        self.param.ue_tx_power = 20
        self.param.ue_height = 1.5
        self.param.ue_aclr = 35
        self.param.ue_acs = 25
        self.param.ue_noise_figure = 9
        self.param.ue_feed_loss = 3
        self.param.dl_attenuation_factor = 0.6
        self.param.dl_sinr_min = -10
        self.param.dl_sinr_max = 30
        self.param.channel_model = "FSPL"
        self.param.line_of_sight_prob = 0.75 # probability of line-of-sight (not for FSPL)
        self.param.noise_temperature = 290
        self.param.BOLTZMANN_CONSTANT = 1.38064852e-23
        
        self.param_ant = ParametersAntennaImt()
        self.param_ant.bs_tx_element_max_g = 10
        self.param_ant.bs_tx_element_phi_3db = 80
        self.param_ant.bs_tx_element_theta_3db = 80
        self.param_ant.bs_tx_element_am = 25
        self.param_ant.bs_tx_element_sla_v = 25
        self.param_ant.bs_tx_n_rows = 16
        self.param_ant.bs_tx_n_columns = 16
        self.param_ant.bs_tx_element_horiz_spacing = 1
        self.param_ant.bs_tx_element_vert_spacing = 1
        self.param_ant.bs_rx_element_max_g = 5
        self.param_ant.bs_rx_element_phi_3db = 65
        self.param_ant.bs_rx_element_theta_3db = 65
        self.param_ant.bs_rx_element_am = 30
        self.param_ant.bs_rx_element_sla_v = 30
        self.param_ant.bs_rx_n_rows = 2
        self.param_ant.bs_rx_n_columns = 2
        self.param_ant.bs_rx_element_horiz_spacing = 0.5
        self.param_ant.bs_rx_element_vert_spacing = 0.5
        self.param_ant.ue_tx_element_max_g = 5
        self.param_ant.ue_tx_element_phi_3db = 65
        self.param_ant.ue_tx_element_theta_3db = 65
        self.param_ant.ue_tx_element_am = 30
        self.param_ant.ue_tx_element_sla_v = 30
        self.param_ant.ue_tx_n_rows = 2
        self.param_ant.ue_tx_n_columns = 1
        self.param_ant.ue_tx_element_horiz_spacing = 0.5
        self.param_ant.ue_tx_element_vert_spacing = 0.5
        self.param_ant.ue_rx_element_max_g = 10
        self.param_ant.ue_rx_element_phi_3db = 90
        self.param_ant.ue_rx_element_theta_3db = 90
        self.param_ant.ue_rx_element_am = 25
        self.param_ant.ue_rx_element_sla_v = 25
        self.param_ant.ue_rx_n_rows = 16
        self.param_ant.ue_rx_n_columns = 16
        self.param_ant.ue_rx_element_horiz_spacing = 1
        self.param_ant.ue_rx_element_vert_spacing = 1
        
        self.simulation = SimulationUplink(self.param, ParametersFss(), self.param_ant)
        self.simulation.initialize()
        
        
    def test_simulation_2bs_4ue(self):
        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param,
                                                                       self.param_ant,
                                                                       self.simulation.topology)
        self.simulation.bs.antenna = np.array([AntennaOmni(1), AntennaOmni(2)])
        
        self.simulation.ue = StationFactory.generate_imt_ue(self.param,
                                                            self.param_ant,
                                                            self.simulation.topology)
        self.simulation.ue.x = np.array([20, 70, 110, 170])
        self.simulation.ue.y = np.array([ 0,  0,   0,   0])
        self.simulation.ue.antenna = np.array([AntennaOmni(11), AntennaOmni(12), AntennaOmni(21), AntennaOmni(22)])
        
        self.simulation.connect_ue_to_bs()
        
        self.simulation.select_ue()
        
        self.simulation.coupling_loss_imt = \
            np.transpose(self.simulation.calculate_coupling_loss(self.simulation.ue, 
                                                                 self.simulation.bs,
                                                                 self.simulation.propagation_imt))
        
        self.simulation.scheduler()
        
        self.simulation.power_control()
        
        self.simulation.calculate_sinr()
        


    def test_beamforming_gains(self):
        eps = 1e-2
        
        # Set scenario
        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param,
                                                                       self.param_ant,
                                                                       self.simulation.topology)
        
        self.simulation.ue = StationFactory.generate_imt_ue(self.param,
                                                            self.param_ant,
                                                            self.simulation.topology)
        self.simulation.ue.x = np.array([50.000, 150.000, 43.301, 25.000])
        self.simulation.ue.y = np.array([ 0.000,   0.000, 25.000, 43.301])
        
        # Physical pointing angles
        self.assertEqual(self.simulation.bs.antenna[0].azimuth,0)
        self.assertEqual(self.simulation.bs.antenna[0].elevation,-10)
        self.assertEqual(self.simulation.bs.antenna[1].azimuth,180)
        self.assertEqual(self.simulation.bs.antenna[0].elevation,-10)
        
        # Change UE pointing
        self.simulation.ue.azimuth = np.array([180, 90, 30, -30])
        self.simulation.ue.elevation = np.array([-30, -15, 15, 30])
        par = self.param_ant.get_antenna_parameters("UE","TX")
        for i in range(self.simulation.ue.num_stations):
            self.simulation.ue.antenna[i] = AntennaBeamformingImt(par, self.simulation.ue.azimuth[i], 
                                                                  self.simulation.ue.elevation[i])
        self.assertEqual(self.simulation.ue.antenna[0].azimuth,180)
        self.assertEqual(self.simulation.ue.antenna[0].elevation,-30)
        self.assertEqual(self.simulation.ue.antenna[1].azimuth,90)
        self.assertEqual(self.simulation.ue.antenna[1].elevation,-15)
        self.assertEqual(self.simulation.ue.antenna[2].azimuth,30)
        self.assertEqual(self.simulation.ue.antenna[2].elevation,15)
        self.assertEqual(self.simulation.ue.antenna[3].azimuth,-30)
        self.assertEqual(self.simulation.ue.antenna[3].elevation,30)
        
        # Simulate connection and selection
        self.simulation.connect_ue_to_bs()
        self.simulation.select_ue()
        # Deactivate some stations for test
        self.simulation.ue.active = [True, False, False, True]
        
        # Test gains
        ref_gain = np.array([[]])
        gain = self.simulation.calculate_gains(self.simulation.bs,self.simulation.ue)
#        npt.assert_allclose(gain,ref_gain,atol=eps)
    
    def test_calculate_coupling_loss(self):
        pass
    
    
    def test_calculate_imt_ul_tput(self):
        pass
    
                
if __name__ == '__main__':
    unittest.main()
    
