# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:06:41 2017

@author: edgar
"""

import numpy as np
import math

from sharc.simulation import Simulation
from sharc.parameters.parameters import Parameters
from sharc.station_factory import StationFactory
from sharc.support.enumerations import StationType

from sharc.propagation.propagation_factory import PropagationFactory


class SimulationDownlink(Simulation):
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
            pass
        else:
            # Execute this piece of code if RLAN generates interference into
            # the other system
            self.calculate_sinr()
            self.calculate_external_interference()
            pass

        self.collect_results(write_to_file, snapshot_number)

    def finalize(self, *args, **kwargs):
        self.notify_observers(source=__name__, results=self.results)

    def power_control(self):
        """
        Apply downlink power control algorithm
        """
        # Currently, the maximum transmit power of the base station is equaly
        # divided among the selected UEs
        total_power = self.parameters.rlan.ap_conducted_power \
                      + self.ap_power_gain
        tx_power = total_power - 10 * math.log10(self.parameters.rlan.ue_k)
        # calculate transmit powers to have a structure such as
        # {ap_1: [pwr_1, pwr_2,...], ...}, where ap_1 is the base station id,
        # pwr_1 is the transmit power from ap_1 to ue_1, pwr_2 is the transmit
        # power from ap_1 to ue_2, etc
        ap_active = np.where(self.ap.active)[0]
        self.ap.tx_power = dict([(ap, tx_power*np.ones(self.parameters.rlan.ue_k)) for ap in ap_active])

        # Update the spectral mask
        if self.adjacent_channel:
            self.ap.spectral_mask.set_mask(power = total_power)

    def calculate_sinr(self):
        """
        Calculates the downlink SINR for each UE.
        """
        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue = self.link[ap]
            self.ue.rx_power[ue] = self.ap.tx_power[ap] - self.parameters.rlan.ap_ohmic_loss \
                                       - self.coupling_loss_rlan[ap,ue] \
                                       - self.parameters.rlan.ue_body_loss \
                                       - self.parameters.rlan.ue_ohmic_loss

            # create a list with base stations that generate interference in ue_list
            ap_interf = [b for b in ap_active if b not in [ap]]

            # calculate intra system interference
            for bi in ap_interf:
                interference = self.ap.tx_power[bi] - self.parameters.rlan.ap_ohmic_loss \
                                 - self.coupling_loss_rlan[bi,ue] \
                                 - self.parameters.rlan.ue_body_loss - self.parameters.rlan.ue_ohmic_loss

                self.ue.rx_interference[ue] = 10*np.log10( \
                    np.power(10, 0.1*self.ue.rx_interference[ue]) + np.power(10, 0.1*interference))

        self.ue.thermal_noise = \
            10*math.log10(self.parameters.rlan.BOLTZMANN_CONSTANT*self.parameters.rlan.noise_temperature*1e3) + \
            10*np.log10(self.ue.bandwidth * 1e6) + \
            self.ue.noise_figure

        self.ue.total_interference = \
            10*np.log10(np.power(10, 0.1*self.ue.rx_interference) + \
                        np.power(10, 0.1*self.ue.thermal_noise))

        self.ue.sinr = self.ue.rx_power - self.ue.total_interference
        self.ue.snr = self.ue.rx_power - self.ue.thermal_noise

    def calculate_sinr_ext(self):
        """
        Calculates the downlink SINR and INR for each UE taking into account the
        interference that is generated by the other system into RLAN system.
        """
        self.coupling_loss_rlan_system = self.calculate_coupling_loss(self.system,
                                                                     self.ue,
                                                                     self.propagation_system,
                                                                     c_channel = self.co_channel)

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        # calculate interference only to active UE's
        ue = np.where(self.ue.active)[0]

        tx_power_sys = self.param_system.tx_power_density + 10*np.log10(self.ue.bandwidth[ue]*1e6) + 30
        self.ue.ext_interference[ue] = tx_power_sys - self.coupling_loss_rlan_system[ue] \
                            - self.parameters.rlan.ue_body_loss - self.parameters.rlan.ue_ohmic_loss

        self.ue.sinr_ext[ue] = self.ue.rx_power[ue] \
            - (10*np.log10(np.power(10, 0.1*self.ue.total_interference[ue]) + np.power(10, 0.1*self.ue.ext_interference[ue])))
        self.ue.inr[ue] = self.ue.ext_interference[ue] - self.ue.thermal_noise[ue]

    def calculate_external_interference(self):
        """
        Calculates interference that RLAN system generates on other system
        """

        if self.co_channel:
            self.coupling_loss_rlan_system = self.calculate_coupling_loss(self.system,
                                                                     self.ap,
                                                                     self.propagation_system) + self.polarization_loss

        if self.adjacent_channel:
            self.coupling_loss_rlan_system_adjacent = self.calculate_coupling_loss(self.system,
                                                                     self.ap,
                                                                     self.propagation_system,
                                                                     c_channel=False) + self.polarization_loss

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the interfered systems bandwidth
        # calculate interference only from active UE's
        rx_interference = 0

        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:

            active_beams = [i for i in range(ap*self.parameters.rlan.ue_k, (ap+1)*self.parameters.rlan.ue_k)]

            if self.co_channel:
                if self.overlapping_bandwidth:
                    acs = 0
                else:
                    acs = self.param_system.adjacent_ch_selectivity

                interference = self.ap.tx_power[ap] - self.parameters.rlan.ap_ohmic_loss \
                             - self.coupling_loss_rlan_system[active_beams]
                weights = self.calculate_bw_weights(self.parameters.rlan.bandwidth,
                                                    self.param_system.bandwidth,
                                                    self.parameters.rlan.ue_k)

                rx_interference += np.sum(weights*np.power(10, 0.1*interference)) / 10**(acs/10.)

            if self.adjacent_channel:

                oob_power = self.ap.spectral_mask.power_calc(self.param_system.frequency,self.system.bandwidth)

                oob_interference = oob_power - self.coupling_loss_rlan_system_adjacent[active_beams[0]] \
                                   + 10*np.log10((self.param_system.bandwidth - self.overlapping_bandwidth)/
                                                 self.param_system.bandwidth)
                                   
                rx_interference += math.pow(10, 0.1*oob_interference)

        self.system.rx_interference = 10*np.log10(rx_interference)
        # calculate N
        self.system.thermal_noise = \
            10*math.log10(self.param_system.BOLTZMANN_CONSTANT* \
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
                self.results.system_dl_interf_power.extend([self.system.rx_interference])

        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue = self.link[ap]
            self.results.rlan_path_loss.extend(self.path_loss_rlan[ap,ue])
            self.results.rlan_coupling_loss.extend(self.coupling_loss_rlan[ap,ue])

            self.results.rlan_ap_antenna_gain.extend(self.rlan_ap_antenna_gain[ap,ue])
            self.results.rlan_ue_antenna_gain.extend(self.rlan_ue_antenna_gain[ap,ue])

            tput = self.calculate_rlan_tput(self.ue.sinr[ue],
                                           self.parameters.rlan.dl_sinr_min,
                                           self.parameters.rlan.dl_sinr_max,
                                           self.parameters.rlan.dl_attenuation_factor)
            self.results.rlan_dl_tput.extend(tput.tolist())

            if self.parameters.rlan.interfered_with:
                tput_ext = self.calculate_rlan_tput(self.ue.sinr_ext[ue],
                                                   self.parameters.rlan.dl_sinr_min,
                                                   self.parameters.rlan.dl_sinr_max,
                                                   self.parameters.rlan.dl_attenuation_factor)
                self.results.rlan_dl_tput_ext.extend(tput_ext.tolist())
                self.results.rlan_dl_sinr_ext.extend(self.ue.sinr_ext[ue].tolist())
                self.results.rlan_dl_inr.extend(self.ue.inr[ue].tolist())

                self.results.system_rlan_antenna_gain.extend(self.system_rlan_antenna_gain[0,ue])
                self.results.rlan_system_antenna_gain.extend(self.rlan_system_antenna_gain[0,ue])
                self.results.rlan_system_path_loss.extend(self.rlan_system_path_loss[0,ue])
                if self.param_system.channel_model == "HDFSS":
                    self.results.rlan_system_build_entry_loss.extend(self.rlan_system_build_entry_loss[0,ue])
                    self.results.rlan_system_diffraction_loss.extend(self.rlan_system_diffraction_loss[0,ue])
            else:
                active_beams = [i for i in range(ap*self.parameters.rlan.ue_k, (ap+1)*self.parameters.rlan.ue_k)]
                self.results.system_rlan_antenna_gain.extend(self.system_rlan_antenna_gain[0,active_beams])
                self.results.rlan_system_antenna_gain.extend(self.rlan_system_antenna_gain[0,active_beams])
                self.results.rlan_system_path_loss.extend(self.rlan_system_path_loss[0,active_beams])
                    
                if self.param_system.channel_model == "HDFSS":
                    self.results.rlan_system_build_entry_loss.extend(self.rlan_system_build_entry_loss[:,ap])
                    self.results.rlan_system_diffraction_loss.extend(self.rlan_system_diffraction_loss[:,ap])

            self.results.rlan_dl_tx_power.extend(self.ap.tx_power[ap].tolist())

            self.results.rlan_dl_sinr.extend(self.ue.sinr[ue].tolist())
            self.results.rlan_dl_snr.extend(self.ue.snr[ue].tolist())

        if write_to_file:
            self.results.write_files(snapshot_number)
            self.notify_observers(source=__name__, results=self.results)

