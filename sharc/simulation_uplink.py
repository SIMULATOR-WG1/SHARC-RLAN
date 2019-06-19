# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:02:35 2017

@author: edgar
"""

import numpy as np
import math

from sharc.simulation import Simulation
from sharc.parameters.parameters import Parameters
from sharc.station_factory import StationFactory
from sharc.support.enumerations import StationType

from sharc.propagation.propagation_factory import PropagationFactory

class SimulationUplink(Simulation):
    """
    Implements the flowchart of simulation downlink method
    """

    def __init__(self, parameters: Parameters, parameter_file: str):
        super().__init__(parameters, parameter_file)

    def snapshot(self, *args, **kwargs):
        write_to_file = kwargs["write_to_file"]
        snapshot_number = kwargs["snapshot_number"]
        seed = kwargs["seed"]

        random_number_gen = np.random.RandomState(seed)

        self.propagation_rlan = PropagationFactory.create_propagation(self.parameters.rlan.channel_model, self.parameters,
                                                                     random_number_gen)
        self.propagation_system = PropagationFactory.create_propagation(self.param_system.channel_model, self.parameters,
                                                                        random_number_gen)

        # In case of hotspots, base stations coordinates have to be calculated
        # on every snapshot. Anyway, let topology decide whether to calculate
        # or not
        self.topology.calculate_coordinates(random_number_gen)

        # Create the base stations (remember that it takes into account the
        # network load factor)
        self.ap = StationFactory.generate_rlan_access_points(self.parameters.rlan,
                                                            self.parameters.antenna_rlan,
                                                            self.topology, random_number_gen)

        # Create the other system (FSS, HAPS, etc...)
        self.system = StationFactory.generate_system(self.parameters, self.topology, random_number_gen)

        # Create RLAN user equipments
        self.ue = StationFactory.generate_rlan_ue(self.parameters.rlan,
                                                 self.parameters.antenna_rlan,
                                                 self.topology, random_number_gen)
        #self.plot_scenario()

        self.connect_ue_to_ap()
        self.select_ue(random_number_gen)

        # Calculate coupling loss after beams are created
        self.coupling_loss_rlan = self.calculate_coupling_loss(self.ap,
                                                              self.ue,
                                                              self.propagation_rlan)
        self.scheduler()
        self.power_control()

        if self.parameters.rlan.interfered_with:
            # Execute this piece of code if the other system generates
            # interference into RLAN
            self.calculate_sinr()
            self.calculate_sinr_ext()
            #self.add_external_interference()
            #self.recalculate_sinr()
            #self.calculate_rlan_degradation()
            pass
        else:
            # Execute this piece of code if RLAN generates interference into
            # the other system
            self.calculate_sinr()
            self.calculate_external_interference()
            #self.calculate_external_degradation()
            pass

        self.collect_results(write_to_file, snapshot_number)


    def power_control(self):
        """
        Apply uplink power control algorithm
        """
        if self.parameters.rlan.rlan_type == "WIFI":
            self.parameters.rlan.ue_tx_power_control == "OFF"
            
        if self.parameters.rlan.rlan_type == "LAA":
            self.parameters.rlan.ue_tx_power_control == "ON"

        if self.parameters.rlan.ue_tx_power_control == "OFF":
            ue_active = np.where(self.ue.active)[0]
            self.ue.tx_power[ue_active] = self.parameters.rlan.ue_p_cmax * np.ones(len(ue_active))
        else:
            ap_active = np.where(self.ap.active)[0]
            for ap in ap_active:
                ue = self.link[ap]
                p_cmax = self.parameters.rlan.ue_p_cmax
                m_pusch = self.num_rb_per_ue
                p_o_pusch = self.parameters.rlan.ue_p_o_pusch
                alpha = self.parameters.rlan.ue_alpha
                cl = self.coupling_loss_rlan[ap,ue] + self.parameters.rlan.ap_ohmic_loss \
                            + self.parameters.rlan.ue_ohmic_loss + self.parameters.rlan.ue_body_loss
                self.ue.tx_power[ue] = np.minimum(p_cmax, 10*np.log10(m_pusch) + p_o_pusch + alpha*cl)
        if self.adjacent_channel: 
            self.ue_power_diff = self.parameters.rlan.ue_p_cmax - self.ue.tx_power


    def calculate_sinr(self):
        """
        Calculates the uplink SINR for each AP.
        """
        # calculate uplink received power for each active AP
        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue = self.link[ap]

            self.ap.rx_power[ap] = self.ue.tx_power[ue]  \
                                        - self.parameters.rlan.ue_ohmic_loss \
                                        - self.parameters.rlan.ue_body_loss \
                                        - self.coupling_loss_rlan[ap,ue] - self.parameters.rlan.ap_ohmic_loss
            # create a list of APs that serve the interfering UEs
            ap_interf = [b for b in ap_active if b not in [ap]]

            # calculate intra system interference
            for bi in ap_interf:
                ui = self.link[bi]
                interference = self.ue.tx_power[ui] - self.parameters.rlan.ue_ohmic_loss  \
                                - self.parameters.rlan.ue_body_loss \
                                - self.coupling_loss_rlan[ap,ui] - self.parameters.rlan.ap_ohmic_loss
                self.ap.rx_interference[ap] = 10*np.log10( \
                    np.power(10, 0.1*self.ap.rx_interference[ap])
                    + np.power(10, 0.1*interference))

            # calculate N
            self.ap.thermal_noise[ap] = \
                10*np.log10(self.parameters.rlan.BOLTZMANN_CONSTANT*self.parameters.rlan.noise_temperature*1e3) + \
                10*np.log10(self.ap.bandwidth[ap] * 1e6) + \
                self.ap.noise_figure[ap]

            # calculate I+N
            self.ap.total_interference[ap] = \
                10*np.log10(np.power(10, 0.1*self.ap.rx_interference[ap]) + \
                            np.power(10, 0.1*self.ap.thermal_noise[ap]))

            # calculate SNR and SINR
            self.ap.sinr[ap] = self.ap.rx_power[ap] - self.ap.total_interference[ap]
            self.ap.snr[ap] = self.ap.rx_power[ap] - self.ap.thermal_noise[ap]


    def calculate_sinr_ext(self):
        """
        Calculates the downlink SINR for each UE taking into account the
        interference that is generated by the other system into RLAN system.
        """
        self.coupling_loss_rlan_system = self.calculate_coupling_loss(self.system,
                                                                     self.ap,
                                                                     self.propagation_system)

        ap_active = np.where(self.ap.active)[0]
        tx_power = self.param_system.tx_power_density + 10*np.log10(self.ap.bandwidth*1e6) + 30
        for ap in ap_active:
            active_beams = [i for i in range(ap*self.parameters.rlan.ue_k, (ap+1)*self.parameters.rlan.ue_k)]
            self.ap.ext_interference[ap] = tx_power[ap] - self.coupling_loss_rlan_system[active_beams] \
                                            - self.parameters.rlan.ap_ohmic_loss

            self.ap.sinr_ext[ap] = self.ap.rx_power[ap] \
                - (10*np.log10(np.power(10, 0.1*self.ap.total_interference[ap]) + np.power(10, 0.1*self.ap.ext_interference[ap])))
            self.ap.inr[ap] = self.ap.ext_interference[ap] - self.ap.thermal_noise[ap]


    def calculate_external_interference(self):
        """
        Calculates interference that RLAN system generates on other system
        """
        if self.co_channel:
            self.coupling_loss_rlan_system = self.calculate_coupling_loss(self.system,
                                                                         self.ue,
                                                                         self.propagation_system)

        if self.adjacent_channel:
              self.coupling_loss_rlan_system_adjacent = self.calculate_coupling_loss(self.system,
                                                                       self.ue,
                                                                       self.propagation_system,
                                                                       c_channel=False)

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        # calculate interference only from active UE's
        rx_interference = 0

        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue = self.link[ap]

            if self.co_channel:
                if self.overlapping_bandwidth:
                    acs = 0
                else:
                    acs = self.param_system.adjacent_ch_selectivity

                interference_ue = self.ue.tx_power[ue] - self.parameters.rlan.ue_ohmic_loss \
                                  - self.parameters.rlan.ue_body_loss \
                                  - self.coupling_loss_rlan_system[ue]
                weights = self.calculate_bw_weights(self.parameters.rlan.bandwidth,
                                                    self.param_system.bandwidth,
                                                    self.parameters.rlan.ue_k)
                rx_interference += np.sum(weights*np.power(10, 0.1*interference_ue)) / 10**(acs/10.)

            if self.adjacent_channel:
                oob_power = self.ue.spectral_mask.power_calc(self.param_system.frequency,self.system.bandwidth)\
                            - self.ue_power_diff[ue]
                oob_interference_array = oob_power - self.coupling_loss_rlan_system_adjacent[ue] \
                                            + 10*np.log10((self.param_system.bandwidth - self.overlapping_bandwidth)/
                                              self.param_system.bandwidth) \
                                            - self.parameters.rlan.ue_body_loss
                rx_interference += np.sum(np.power(10,0.1*oob_interference_array))

        self.system.rx_interference = 10*np.log10(rx_interference)
        # calculate N
        self.system.thermal_noise = \
            10*np.log10(self.param_system.BOLTZMANN_CONSTANT* \
                          self.system.noise_temperature*1e3) + \
                          10*math.log10(self.param_system.bandwidth * 1e6)

        # calculate INR at the system
        self.system.inr = np.array([self.system.rx_interference - self.system.thermal_noise])

        # Calculate PFD at the system
        if self.system.station_type is StationType.RAS:
            self.system.pfd = 10*np.log10(10**(self.system.rx_interference/10)/self.system.antenna[0].effective_area)


    def collect_results(self, write_to_file: bool, snapshot_number: int):
        if not self.parameters.rlan.interfered_with and np.any(self.ap.active):
            self.results.system_inr.extend(self.system.inr.tolist())
            self.results.system_inr_scaled.extend([self.system.inr + 10*math.log10(self.param_system.inr_scaling)])
            if self.system.station_type is StationType.RAS:
                self.results.system_pfd.extend([self.system.pfd])
                self.results.system_ul_interf_power.extend([self.system.rx_interference])

        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue = self.link[ap]
            self.results.rlan_path_loss.extend(self.path_loss_rlan[ap,ue])
            self.results.rlan_coupling_loss.extend(self.coupling_loss_rlan[ap,ue])

            self.results.rlan_ap_antenna_gain.extend(self.rlan_ap_antenna_gain[ap,ue])
            self.results.rlan_ue_antenna_gain.extend(self.rlan_ue_antenna_gain[ap,ue])

            tput = self.calculate_rlan_tput(self.ap.sinr[ap],
                                           self.parameters.rlan.ul_sinr_min,
                                           self.parameters.rlan.ul_sinr_max,
                                           self.parameters.rlan.ul_attenuation_factor)
            self.results.rlan_ul_tput.extend(tput.tolist())

            if self.parameters.rlan.interfered_with:
                tput_ext = self.calculate_rlan_tput(self.ap.sinr_ext[ap],
                                                      self.parameters.rlan.ul_sinr_min,
                                                      self.parameters.rlan.ul_sinr_max,
                                                      self.parameters.rlan.ul_attenuation_factor)
                self.results.rlan_ul_tput_ext.extend(tput_ext.tolist())
                self.results.rlan_ul_sinr_ext.extend(self.ap.sinr_ext[ap].tolist())
                self.results.rlan_ul_inr.extend(self.ap.inr[ap].tolist())

                active_beams = [i for i in range(ap*self.parameters.rlan.ue_k, (ap+1)*self.parameters.rlan.ue_k)]
                self.results.system_rlan_antenna_gain.extend(self.system_rlan_antenna_gain[0,active_beams])
                self.results.rlan_system_antenna_gain.extend(self.rlan_system_antenna_gain[0,active_beams])
                self.results.rlan_system_path_loss.extend(self.rlan_system_path_loss[0,active_beams])
                if self.param_system.channel_model == "HDFSS":
                    self.results.rlan_system_build_entry_loss.extend(self.rlan_system_build_entry_loss[:,ap])
                    self.results.rlan_system_diffraction_loss.extend(self.rlan_system_diffraction_loss[:,ap])
            else:
                self.results.system_rlan_antenna_gain.extend(self.system_rlan_antenna_gain[0,ue])
                self.results.rlan_system_antenna_gain.extend(self.rlan_system_antenna_gain[0,ue])
                self.results.rlan_system_path_loss.extend(self.rlan_system_path_loss[0,ue])
                if self.param_system.channel_model == "HDFSS":
                    self.results.rlan_system_build_entry_loss.extend(self.rlan_system_build_entry_loss[:,ue])
                    self.results.rlan_system_diffraction_loss.extend(self.rlan_system_diffraction_loss[:,ue])

            self.results.rlan_ul_tx_power.extend(self.ue.tx_power[ue].tolist())
            rlan_ul_tx_power_density = 10*np.log10(np.power(10, 0.1*self.ue.tx_power[ue])/(self.num_rb_per_ue*self.parameters.rlan.rb_bandwidth*1e6))
            self.results.rlan_ul_tx_power_density.extend(rlan_ul_tx_power_density.tolist())
            self.results.rlan_ul_sinr.extend(self.ap.sinr[ap].tolist())
            self.results.rlan_ul_snr.extend(self.ap.snr[ap].tolist())

        if write_to_file:
            self.results.write_files(snapshot_number)
            self.notify_observers(source=__name__, results=self.results)



