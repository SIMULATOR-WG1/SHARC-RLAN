# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:04:03 2017

@author: edgar
"""

from abc import ABC, abstractmethod
from sharc.support.observable import Observable

import numpy as np
import math
import sys
import matplotlib.pyplot as plt

from sharc.support.enumerations import StationType
from sharc.topology.topology_factory import TopologyFactory
from sharc.parameters.parameters import Parameters
from sharc.propagation.propagation import Propagation
from sharc.station_manager import StationManager
from sharc.results import Results
from sharc.propagation.propagation_factory import PropagationFactory


class Simulation(ABC, Observable):

    def __init__(self, parameters: Parameters, parameter_file: str):
        ABC.__init__(self)
        Observable.__init__(self)

        self.parameters = parameters
        self.parameters_filename = parameter_file

        if self.parameters.general.system == "FSS_SS":
            self.param_system = self.parameters.fss_ss
        elif self.parameters.general.system == "FSS_ES":
            self.param_system = self.parameters.fss_es
        elif self.parameters.general.system == "AMT_EMBRAER":
            self.param_system = self.parameters.amt_embraer
        elif self.parameters.general.system == "FS":
            self.param_system = self.parameters.fs
        elif self.parameters.general.system == "HAPS":
            self.param_system = self.parameters.haps
        elif self.parameters.general.system == "RNS":
            self.param_system = self.parameters.rns
        elif self.parameters.general.system == "RAS":
            self.param_system = self.parameters.ras
            
        self.wrap_around_enabled = self.parameters.rlan.wrap_around and \
                                  (self.parameters.rlan.topology == 'MACROCELL' \
                                   or self.parameters.rlan.topology == 'HOTSPOT') and \
                                   self.parameters.rlan.num_clusters == 1

        self.co_channel = self.parameters.general.enable_cochannel
        self.adjacent_channel = self.parameters.general.enable_adjacent_channel

        self.topology = TopologyFactory.createTopology(self.parameters)

        self.ap_power_gain = 0
        self.ue_power_gain = 0

        self.rlan_ap_antenna_gain = list()
        self.rlan_ue_antenna_gain = list()
        self.system_rlan_antenna_gain = list()
        self.rlan_system_antenna_gain = list()
        self.rlan_system_path_loss = list()
        self.rlan_system_build_entry_loss = list()
        self.rlan_system_diffraction_loss = list()

        self.path_loss_rlan = np.empty(0)
        self.coupling_loss_rlan = np.empty(0)
        self.coupling_loss_rlan_system = np.empty(0)
        self.coupling_loss_rlan_system_adjacent = np.empty(0)

        self.ap_to_ue_d_2D = np.empty(0)
        self.ap_to_ue_d_3D = np.empty(0)
        self.ap_to_ue_phi = np.empty(0)
        self.ap_to_ue_theta = np.empty(0)
        self.ap_to_ue_beam_rbs = np.empty(0)

        self.ue = np.empty(0)
        self.ap = np.empty(0)
        self.system = np.empty(0)

        self.link = dict()

        self.num_rb_per_ap = 0
        self.num_rb_per_ue = 0

        self.results = None

        rlan_min_freq = self.parameters.rlan.frequency - self.parameters.rlan.bandwidth / 2
        rlan_max_freq = self.parameters.rlan.frequency + self.parameters.rlan.bandwidth / 2
        system_min_freq = self.param_system.frequency - self.param_system.bandwidth / 2
        system_max_freq = self.param_system.frequency + self.param_system.bandwidth / 2

        max_min_freq = np.maximum(rlan_min_freq, system_min_freq)
        min_max_freq = np.minimum(rlan_max_freq, system_max_freq)

        self.overlapping_bandwidth = min_max_freq - max_min_freq
        if self.overlapping_bandwidth < 0:
            self.overlapping_bandwidth = 0

        if (self.overlapping_bandwidth == self.param_system.bandwidth and
            not self.parameters.rlan.interfered_with) or \
           (self.overlapping_bandwidth == self.parameters.rlan.bandwidth and
            self.parameters.rlan.interfered_with):

            self.adjacent_channel = False

        self.propagation_rlan = None
        self.propagation_system = None

    def add_observer_list(self, observers: list):
        for o in observers:
            self.add_observer(o)

    def initialize(self, *args, **kwargs):
        """
        This method is executed only once to initialize the simulation variables.
        """

        self.topology.calculate_coordinates()
        num_ap = self.topology.num_access_points
        num_ue = num_ap*self.parameters.rlan.ue_k*self.parameters.rlan.ue_k_m

        self.ap_power_gain = 10*math.log10(self.parameters.antenna_rlan.ap_tx_n_rows*
                                           self.parameters.antenna_rlan.ap_tx_n_columns)
        self.ue_power_gain = 10*math.log10(self.parameters.antenna_rlan.ue_tx_n_rows*
                                           self.parameters.antenna_rlan.ue_tx_n_columns)
        self.rlan_ap_antenna_gain = list()
        self.rlan_ue_antenna_gain = list()
        self.path_loss_rlan = np.empty([num_ap, num_ue])
        self.coupling_loss_rlan = np.empty([num_ap, num_ue])
        self.coupling_loss_rlan_system = np.empty(num_ue)

        self.ap_to_ue_phi = np.empty([num_ap, num_ue])
        self.ap_to_ue_theta = np.empty([num_ap, num_ue])
        self.ap_to_ue_beam_rbs = -1.0*np.ones(num_ue, dtype=int)

        self.ue = np.empty(num_ue)
        self.ap = np.empty(num_ap)
        self.system = np.empty(1)

        # this attribute indicates the list of UE's that are connected to each
        # Access point. The position the the list indicates the resource block
        # group that is allocated to the given UE
        self.link = dict([(ap,list()) for ap in range(num_ap)])

        # calculates the number of RB per AP
        self.num_rb_per_ap = math.trunc((1-self.parameters.rlan.guard_band_ratio)* \
                            self.parameters.rlan.bandwidth /self.parameters.rlan.rb_bandwidth)
        # calculates the number of RB per UE on a given AP
        self.num_rb_per_ue = math.trunc(self.num_rb_per_ap/self.parameters.rlan.ue_k)

        self.results = Results(self.parameters_filename, self.parameters.general.overwrite_output)
        
        if self.parameters.general.system == 'RAS':
            self.polarization_loss = 0.0
        else:
            self.polarization_loss = 3.0

    def finalize(self, *args, **kwargs):
        """
        Finalizes the simulation (collect final results, etc...)
        """
        snapshot_number = kwargs["snapshot_number"]
        self.results.write_files(snapshot_number)

    def calculate_coupling_loss(self,
                                station_a: StationManager,
                                station_b: StationManager,
                                propagation: Propagation,
                                c_channel = True) -> np.array:
        """
        Calculates the path coupling loss from each station_a to all station_b.
        Result is returned as a numpy array with dimensions num_a x num_b
        TODO: calculate coupling loss between activa stations only
        """
        if station_a.station_type is StationType.FSS_SS or \
           station_a.station_type is StationType.HAPS or \
           station_a.station_type is StationType.RNS:
            elevation_angles = station_b.get_elevation_angle(station_a, self.param_system)
        elif station_a.station_type is StationType.RLAN_AP and \
             station_b.station_type is StationType.RLAN_UE and \
             self.parameters.rlan.topology == "INDOOR":
            elevation_angles = np.transpose(station_b.get_elevation(station_a))
        elif station_a.station_type is StationType.FSS_ES or \
            station_a.station_type is StationType.RAS or \
            station_a.station_type is StationType.AMT_EMBRAER:
            elevation_angles = station_b.get_elevation(station_a)
        else:
            elevation_angles = None

        if station_a.station_type is StationType.FSS_SS or \
           station_a.station_type is StationType.FSS_ES or \
           station_a.station_type is StationType.AMT_EMBRAER or \
           station_a.station_type is StationType.HAPS or \
           station_a.station_type is StationType.FS or \
           station_a.station_type is StationType.RNS or \
           station_a.station_type is StationType.RAS:
            # Calculate distance from transmitters to receivers. The result is a
            # num_station_a x num_station_b
            d_2D = station_a.get_distance_to(station_b)
            d_3D = station_a.get_3d_distance_to(station_b)
            
            if self.parameters.rlan.interfered_with:
                freq = self.param_system.frequency
            else:
                freq = self.parameters.rlan.frequency

            if station_b.station_type is StationType.RLAN_UE:
                # define antenna gains
                gain_a = self.calculate_gains(station_a, station_b)
                gain_b = np.transpose(self.calculate_gains(station_b, station_a, c_channel))
                sectors_in_node=1

            else:
                # define antenna gains
                gain_a = np.repeat(self.calculate_gains(station_a, station_b), self.parameters.rlan.ue_k, 1)
                gain_b = np.transpose(self.calculate_gains(station_b, station_a, c_channel))
                sectors_in_node = self.parameters.rlan.ue_k

            if self.parameters.rlan.interfered_with:
                earth_to_space = False
                single_entry = True
            else:
                earth_to_space = True
                single_entry = False

            if station_a.station_type is StationType.FSS_SS or \
               station_a.station_type is StationType.HAPS or \
               station_a.station_type is StationType.RNS:
                path_loss = propagation.get_loss(distance_3D=d_3D,
                                             frequency=freq*np.ones(d_3D.shape),
                                             indoor_stations=np.tile(station_b.indoor, (station_a.num_stations, 1)),
                                             elevation=elevation_angles, sat_params = self.param_system,
                                             earth_to_space = earth_to_space, earth_station_antenna_gain=gain_b,
                                             single_entry=single_entry, number_of_sectors=sectors_in_node)
            else:
                path_loss = propagation.get_loss(distance_3D=d_3D,
                                             frequency=freq*np.ones(d_3D.shape),
                                             indoor_stations=np.tile(station_b.indoor, (station_a.num_stations, 1)),
                                             elevation=elevation_angles, es_params=self.param_system,
                                             tx_gain = gain_a, rx_gain = gain_b, number_of_sectors=sectors_in_node,
                                             rlan_sta_type=station_b.station_type,
                                             rlan_x=station_b.x,
                                             rlan_y=station_b.y,
                                             rlan_z=station_b.height,
                                             es_x=station_a.x,
                                             es_y=station_a.y,
                                             es_z=station_a.height)
                if self.param_system.channel_model == "HDFSS":
                    self.rlan_system_build_entry_loss = path_loss[1]
                    self.rlan_system_diffraction_loss = path_loss[2]
                    path_loss = path_loss[0]

            self.system_rlan_antenna_gain = gain_a
            self.rlan_system_antenna_gain = gain_b
            self.rlan_system_path_loss = path_loss
        # RLAN <-> RLAN
        else:
            d_2D = self.ap_to_ue_d_2D
            d_3D = self.ap_to_ue_d_3D
            freq = self.parameters.rlan.frequency
            
            path_loss = propagation.get_loss(distance_3D=d_3D,
                                             distance_2D=d_2D,
                                             frequency=self.parameters.rlan.frequency*np.ones(d_2D.shape),
                                             indoor_stations=np.tile(station_b.indoor, (station_a.num_stations, 1)),
                                             ap_height=station_a.height,
                                             ue_height=station_b.height,
                                             elevation=elevation_angles,
                                             shadowing=self.parameters.rlan.shadowing,
                                             line_of_sight_prob=self.parameters.rlan.line_of_sight_prob)
            # define antenna gains
            gain_a = self.calculate_gains(station_a, station_b)
            gain_b = np.transpose(self.calculate_gains(station_b, station_a))

            # collect RLAN AP and UE antenna gain samples
            self.path_loss_rlan = path_loss
            self.rlan_ap_antenna_gain = gain_a
            self.rlan_ue_antenna_gain = gain_b

        # calculate coupling loss
        coupling_loss = np.squeeze(path_loss - gain_a - gain_b)

        return coupling_loss

    def connect_ue_to_ap(self):
        """
        Link the UE's to the serving AP. It is assumed that each group of K*M
        user equipments are distributed and pointed to a certain base station
        according to the decisions taken at TG 5/1 meeting
        """
        num_ue_per_ap = self.parameters.rlan.ue_k*self.parameters.rlan.ue_k_m
        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue_list = [i for i in range(ap*num_ue_per_ap, ap*num_ue_per_ap + num_ue_per_ap)]
            self.link[ap] = ue_list

    def select_ue(self, random_number_gen: np.random.RandomState):
        """
        Select K UEs randomly from all the UEs linked to one AP as “chosen”
        UEs. These K “chosen” UEs will be scheduled during this snapshot.
        """
        if self.wrap_around_enabled:
            self.ap_to_ue_d_2D, self.ap_to_ue_d_3D, self.ap_to_ue_phi, self.ap_to_ue_theta = \
                self.ap.get_dist_angles_wrap_around(self.ue)
        else:
            self.ap_to_ue_d_2D = self.ap.get_distance_to(self.ue)
            self.ap_to_ue_d_3D = self.ap.get_3d_distance_to(self.ue)
            self.ap_to_ue_phi, self.ap_to_ue_theta = self.ap.get_pointing_vector_to(self.ue)

        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            # select K UE's among the ones that are connected to AP
            random_number_gen.shuffle(self.link[ap])
            K = self.parameters.rlan.ue_k
            del self.link[ap][K:]
            # Activate the selected UE's and create beams
            if self.ap.active[ap]:
                self.ue.active[self.link[ap]] = np.ones(K, dtype=bool)
                for ue in self.link[ap]:
                    # add beam to AP antennas
                    self.ap.antenna[ap].add_beam(self.ap_to_ue_phi[ap,ue],
                                             self.ap_to_ue_theta[ap,ue])
                    # add beam to UE antennas
                    self.ue.antenna[ue].add_beam(self.ap_to_ue_phi[ap,ue] - 180,
                                             180 - self.ap_to_ue_theta[ap,ue])
                    # set beam resource block group
                    self.ap_to_ue_beam_rbs[ue] = len(self.ap.antenna[ap].beams_list) - 1


    def scheduler(self):
        """
        This scheduler divides the available resource blocks among UE's for
        a given AP
        """
        ap_active = np.where(self.ap.active)[0]
        for ap in ap_active:
            ue = self.link[ap]
            self.ap.bandwidth[ap] = self.num_rb_per_ue*self.parameters.rlan.rb_bandwidth
            self.ue.bandwidth[ue] = self.num_rb_per_ue*self.parameters.rlan.rb_bandwidth

    def calculate_gains(self,
                        station_1: StationManager,
                        station_2: StationManager,
                        c_channel = True) -> np.array:
        """
        Calculates the gains of antennas in station_1 in the direction of
        station_2
        """
        station_1_active = np.where(station_1.active)[0]
        station_2_active = np.where(station_2.active)[0]

        # Initialize variables (phi, theta, beams_idx)
        if(station_1.station_type is StationType.RLAN_AP):
            if(station_2.station_type is StationType.RLAN_UE):
                phi = self.ap_to_ue_phi
                theta = self.ap_to_ue_theta
                beams_idx = self.ap_to_ue_beam_rbs[station_2_active]
            elif(station_2.station_type is StationType.FSS_SS or \
                 station_2.station_type is StationType.FSS_ES or \
                 station_2.station_type is StationType.AMT_EMBRAER or \
                 station_2.station_type is StationType.HAPS or \
                 station_2.station_type is StationType.FS or \
                 station_2.station_type is StationType.RNS or \
                 station_2.station_type is StationType.RAS):
                phi, theta = station_1.get_pointing_vector_to(station_2)
                phi = np.repeat(phi,self.parameters.rlan.ue_k,0)
                theta = np.repeat(theta,self.parameters.rlan.ue_k,0)
                beams_idx = np.tile(np.arange(self.parameters.rlan.ue_k),self.ap.num_stations)

        elif(station_1.station_type is StationType.RLAN_UE):
            phi, theta = station_1.get_pointing_vector_to(station_2)
            beams_idx = np.zeros(len(station_2_active),dtype=int)

        elif(station_1.station_type is StationType.FSS_SS or \
             station_1.station_type is StationType.FSS_ES or \
             station_1.station_type is StationType.AMT_EMBRAER or \
             station_1.station_type is StationType.HAPS or \
             station_1.station_type is StationType.FS or \
             station_1.station_type is StationType.RNS or \
             station_1.station_type is StationType.RAS):
            phi, theta = station_1.get_pointing_vector_to(station_2)
            beams_idx = np.zeros(len(station_2_active),dtype=int)

        # Calculate gains
        gains = np.zeros(phi.shape)
        if (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.FSS_SS) or \
           (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.FSS_ES) or \
           (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.AMT_EMBRAER) or \
           (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.HAPS) or \
           (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.FS) or \
           (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.RNS) or \
           (station_1.station_type is StationType.RLAN_AP and station_2.station_type is StationType.RAS):
            for k in station_1_active:
                for b in range(k*self.parameters.rlan.ue_k,(k+1)*self.parameters.rlan.ue_k):
                    gains[b,station_2_active] = station_1.antenna[k].calculate_gain(phi_vec=phi[b,station_2_active],
                                                                            theta_vec=theta[b,station_2_active],
                                                                            beams_l=np.array([beams_idx[b]]),
                                                                            co_channel=c_channel)

        elif (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.FSS_SS) or \
             (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.FSS_ES) or \
             (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.AMT_EMBRAER) or \
             (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.HAPS) or \
             (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.FS) or \
             (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.RNS) or \
             (station_1.station_type is StationType.RLAN_UE and station_2.station_type is StationType.RAS):
               for k in station_1_active:
                   gains[k,station_2_active] = station_1.antenna[k].calculate_gain(phi_vec=phi[k,station_2_active],
                                                                            theta_vec=theta[k,station_2_active],
                                                                            beams_l=beams_idx,
                                                                            co_channel=c_channel)

        elif station_1.station_type is StationType.RNS:
            gains[0,station_2_active] = station_1.antenna[0].calculate_gain(phi_vec = phi[0,station_2_active],
                                                                            theta_vec = theta[0,station_2_active])

        elif station_1.station_type is StationType.FSS_SS or \
             station_1.station_type is StationType.FSS_ES or \
             station_1.station_type is StationType.AMT_EMBRAER or \
             station_1.station_type is StationType.HAPS or \
             station_1.station_type is StationType.FS or \
             station_1.station_type is StationType.RAS:

            off_axis_angle = station_1.get_off_axis_angle(station_2)
            distance = station_1.get_distance_to(station_2)
            theta = np.degrees(np.arctan((station_1.height - station_2.height)/distance)) + station_1.elevation
            gains[0,station_2_active] = station_1.antenna[0].calculate_gain(off_axis_angle_vec=off_axis_angle[0,station_2_active],
                                                                            theta_vec=theta[0,station_2_active])
        else: # for RLAN <-> RLAN
            for k in station_1_active:
                gains[k,station_2_active] = station_1.antenna[k].calculate_gain(phi_vec=phi[k,station_2_active],
                                                                            theta_vec=theta[k,station_2_active],
                                                                            beams_l=beams_idx)

        return gains

    def calculate_rlan_tput(self,
                           sinr: np.array,
                           sinr_min: float,
                           sinr_max: float,
                           attenuation_factor: float) -> np.array:
        tput_min = 0
        tput_max = attenuation_factor*math.log2(1+math.pow(10, 0.1*sinr_max))

        tput = attenuation_factor*np.log2(1+np.power(10, 0.1*sinr))

        id_min = np.where(sinr < sinr_min)[0]
        id_max = np.where(sinr > sinr_max)[0]

        if len(id_min) > 0:
            tput[id_min] = tput_min
        if len(id_max) > 0:
            tput[id_max] = tput_max

        return tput

    def calculate_bw_weights(self, bw_rlan: float, bw_sys: float, ue_k: int) -> np.array:
        """
        Calculates the weight that each resource block group of RLAN base stations
        will have when estimating the interference to other systems based on
        the bandwidths of both systems.

        Parameters
        ----------
            bw_rlan : bandwidth of RLAN system
            bw_sys : bandwidth of other system
            ue_k : number of UE's allocated to each RLAN base station; it also
                corresponds to the number of resource block groups

        Returns
        -------
            K-dimentional array of weights
        """

        if bw_rlan <= bw_sys:
            weights = np.ones(ue_k)

        elif bw_rlan > bw_sys:
            weights = np.zeros(ue_k)

            bw_per_rbg = bw_rlan / ue_k

            # number of resource block groups that will have weight equal to 1
            rb_ones = math.floor( bw_sys / bw_per_rbg )

            # weight of the rbg that will generate partial interference
            rb_partial = np.mod( bw_sys, bw_per_rbg ) / bw_per_rbg

            # assign value to weight array
            weights[:rb_ones] = 1
            weights[rb_ones] = rb_partial

        return weights

    def plot_scenario(self):
        fig = plt.figure(figsize=(8,8), facecolor='w', edgecolor='k')
        ax = fig.gca()

        # Plot network topology
        self.topology.plot(ax)

        # Plot user equipments
        ax.scatter(self.ue.x, self.ue.y, color='r', edgecolor="w", linewidth=0.5, label="UE")

        # Plot UE's azimuth
        d = 0.1 * self.topology.cell_radius
        for i in range(len(self.ue.x)):
            plt.plot([self.ue.x[i], self.ue.x[i] + d*math.cos(math.radians(self.ue.azimuth[i]))],
                     [self.ue.y[i], self.ue.y[i] + d*math.sin(math.radians(self.ue.azimuth[i]))],
                     'r-')

        plt.axis('image')
        plt.title("Simulation scenario")
        plt.xlabel("x-coordinate [m]")
        plt.ylabel("y-coordinate [m]")
        plt.legend(loc="upper left", scatterpoints=1)
        plt.tight_layout()
        plt.show()

        if self.parameters.rlan.topology == "INDOOR":
            fig = plt.figure(figsize=(8,8), facecolor='w', edgecolor='k')
            ax = fig.gca()

            # Plot network topology
            self.topology.plot(ax,top_view=False)

            # Plot user equipments
            ax.scatter(self.ue.x, self.ue.height, color='r', edgecolor="w", linewidth=0.5, label="UE")

            plt.title("Simulation scenario: side view")
            plt.xlabel("x-coordinate [m]")
            plt.ylabel("z-coordinate [m]")
            plt.legend(loc="upper left", scatterpoints=1)
            plt.tight_layout()
            plt.show()
        
#        sys.exit(0)

    @abstractmethod
    def snapshot(self, *args, **kwargs):
        """
        Performs a single snapshot.
        """
        pass

    @abstractmethod
    def power_control(self):
        """
        Apply downlink power control algorithm
        """

    @abstractmethod
    def collect_results(self, *args, **kwargs):
        """
        Collects results.
        """
        pass
