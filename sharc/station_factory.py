# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:37:32 2017

@author: edgar
"""

import numpy as np
import sys
import math

from sharc.support.enumerations import StationType
from sharc.parameters.parameters import Parameters
from sharc.parameters.parameters_rlan import ParametersRlan
from sharc.parameters.parameters_antenna_rlan import ParametersAntennaRlan
from sharc.parameters.parameters_fs import ParametersFs
from sharc.parameters.parameters_fss_ss import ParametersFssSs
from sharc.parameters.parameters_fss_es import ParametersFssEs
from sharc.parameters.parameters_amt_gs import ParametersAmtGs
from sharc.parameters.parameters_rdr_gs import ParametersRdrGs
from sharc.parameters.parameters_amax_bs import ParametersAmaxBs
from sharc.parameters.parameters_amax_cpe import ParametersAmaxCpe
from sharc.parameters.parameters_haps import ParametersHaps
from sharc.parameters.parameters_rns import ParametersRns
from sharc.parameters.parameters_ras import ParametersRas
from sharc.station_manager import StationManager
from sharc.spectral_mask_rlan import SpectralMaskRlan
from sharc.spectral_mask_laa import SpectralMaskLaa
from sharc.antenna.antenna import Antenna
from sharc.antenna.antenna_fss_ss import AntennaFssSs
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.antenna.antenna_f699 import AntennaF699
from sharc.antenna.antenna_f1891 import AntennaF1891
from sharc.antenna.antenna_m1466 import AntennaM1466
from sharc.antenna.antenna_s465 import AntennaS465
from sharc.antenna.antenna_modified_s465 import AntennaModifiedS465
from sharc.antenna.antenna_s580 import AntennaS580
from sharc.antenna.antenna_r1652_1 import AntennaRadar
from sharc.antenna.antenna_s580_rlan import AntennaS580_rlan
from sharc.antenna.antenna_s672 import AntennaS672
from sharc.antenna.antenna_s1528 import AntennaS1528
from sharc.antenna.antenna_s1855 import AntennaS1855
from sharc.antenna.antenna_sa509 import AntennaSA509
from sharc.antenna.antenna_element_aeromax_f1336 import AntennaElementAeromaxF1336
from sharc.antenna.antenna_beamforming_rlan import AntennaBeamformingRlan
from sharc.topology.topology import Topology
from sharc.topology.topology_macrocell import TopologyMacrocell
from sharc.spectral_mask_3gpp import SpectralMask3Gpp


class StationFactory(object):

    @staticmethod
    def generate_rlan_access_points(param: ParametersRlan,
                                   param_ant: ParametersAntennaRlan,
                                   topology: Topology,
                                   random_number_gen: np.random.RandomState):
        num_ap = topology.num_access_points
        rlan_access_points = StationManager(num_ap)
        rlan_access_points.station_type = StationType.RLAN_AP
        # now we set the coordinates
        rlan_access_points.x = topology.x
        rlan_access_points.y = topology.y
        rlan_access_points.azimuth = topology.azimuth
        rlan_access_points.elevation = topology.elevation
        rlan_access_points.indoor = random_number_gen.random_sample(num_ap) >= 0
        if param.topology == 'INDOOR':
            rlan_access_points.height = topology.height
        else:
            rlan_access_points.height = param.ap_height*np.ones(num_ap)

        rlan_access_points.active = random_number_gen.rand(num_ap) < param.ap_load_probability
        rlan_access_points.tx_power = param.ap_conducted_power*np.ones(num_ap)
        rlan_access_points.rx_power = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])
        rlan_access_points.rx_interference = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])
        rlan_access_points.ext_interference = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])
        rlan_access_points.total_interference = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])

        rlan_access_points.snr = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])
        rlan_access_points.sinr = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])
        rlan_access_points.sinr_ext = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])
        rlan_access_points.inr = dict([(ap, -500 * np.ones(param.ue_k)) for ap in range(num_ap)])

        rlan_access_points.antenna = np.empty(num_ap, dtype=AntennaBeamformingRlan)
        par = param_ant.get_antenna_parameters("AP", "RX")

        for i in range(num_ap):
            rlan_access_points.antenna[i] = \
            AntennaBeamformingRlan(par, rlan_access_points.azimuth[i],\
                                  rlan_access_points.elevation[i], param.rlan_type)

        #rlan_access_points.antenna = [AntennaOmni(0) for ap in range(num_ap)]
        rlan_access_points.bandwidth = param.bandwidth*np.ones(num_ap)
        rlan_access_points.center_freq = param.frequency*np.ones(num_ap)
        rlan_access_points.noise_figure = param.ap_noise_figure*np.ones(num_ap)
        rlan_access_points.thermal_noise = -500*np.ones(num_ap)

        if param.spectral_mask == "RLAN":
            rlan_access_points.spectral_mask = SpectralMaskRlan(StationType.RLAN_AP,param.frequency,\
                                                              param.bandwidth,scenario = param.topology)
        elif param.spectral_mask == "LAA":
            rlan_access_points.spectral_mask = SpectralMaskLaa(StationType.RLAN_AP,param.frequency,\
                                                               param.bandwidth)
            
        if param.topology == 'MACROCELL' or param.topology == 'HOTSPOT':
            rlan_access_points.intesite_dist = param.intersite_distance

        return rlan_access_points

    @staticmethod
    def generate_rlan_ue(param: ParametersRlan,
                        param_ant: ParametersAntennaRlan,
                        topology: Topology,
                        random_number_gen: np.random.RandomState)-> StationManager:

        if param.topology == "INDOOR":
            return StationFactory.generate_rlan_ue_indoor(param, param_ant, random_number_gen, topology)
        else:
            return StationFactory.generate_rlan_ue_outdoor(param, param_ant, random_number_gen, topology)


    @staticmethod
    def generate_rlan_ue_outdoor(param: ParametersRlan,
                                param_ant: ParametersAntennaRlan,
                                random_number_gen: np.random.RandomState,
                                topology: Topology) -> StationManager:
        num_ap = topology.num_access_points
        num_ue_per_ap = param.ue_k*param.ue_k_m

        num_ue = num_ap * num_ue_per_ap

        rlan_ue = StationManager(num_ue)
        rlan_ue.station_type = StationType.RLAN_UE

        ue_x = list()
        ue_y = list()

        # Calculate UE pointing
        azimuth_range = (-180, 180)
        azimuth = (azimuth_range[1] - azimuth_range[0])*random_number_gen.random_sample(num_ue) + azimuth_range[0]
        # Remove the randomness from azimuth and you will have a perfect pointing
        elevation_range = (-90, 90)
        elevation = (elevation_range[1] - elevation_range[0])*random_number_gen.random_sample(num_ue) + \
                    elevation_range[0]

        if param.ue_distribution_type.upper() == "UNIFORM":

            if not (type(topology) is TopologyMacrocell):
                sys.stderr.write("ERROR\nUniform UE distribution is currently supported only with Macrocell topology")
                sys.exit(1)

            [ue_x, ue_y, theta, distance] = StationFactory.get_random_position(num_ue, topology, random_number_gen,
                                                                               param.minimum_separation_distance_ap_ue )
            psi = np.degrees(np.arctan((param.ap_height - param.ue_height) / distance))

            rlan_ue.azimuth = (azimuth + theta + np.pi/2)
            rlan_ue.elevation = elevation + psi


        elif param.ue_distribution_type.upper() == "ANGLE_AND_DISTANCE":
            # The Rayleigh and Normal distribution parameters (mean, scale and cutoff)
            # were agreed in TG 5/1 meeting (May 2017).

            if param.ue_distribution_distance.upper() == "RAYLEIGH":
                # For the distance between UE and AP, it is desired that 99% of UE's
                # are located inside the [soft] cell edge, i.e. Prob(d<d_edge) = 99%.
                # Since the distance is modeled by a random variable with Rayleigh
                # distribution, we use the quantile function to find that
                # sigma = distance/3.0345. So we always distibute UE's in order to meet
                # the requirement Prob(d<d_edge) = 99% for a given cell radius.
                radius_scale = topology.cell_radius / 3.0345
                radius = random_number_gen.rayleigh(radius_scale, num_ue)
            elif param.ue_distribution_distance.upper() == "UNIFORM":
                radius = topology.cell_radius * random_number_gen.random_sample(num_ue)
            else:
                sys.stderr.write("ERROR\nInvalid UE distance distribution: " + param.ue_distribution_distance)
                sys.exit(1)

            if param.ue_distribution_azimuth.upper() == "NORMAL":
                # In case of the angles, we generate N times the number of UE's because
                # the angle cutoff will discard 5% of the terminals whose angle is
                # outside the angular sector defined by [-60, 60]. So, N = 1.4 seems to
                # be a safe choice.
                N = 1.4
                angle_scale = 30
                angle_mean = 0
                angle_n = random_number_gen.normal(angle_mean, angle_scale, int(N * num_ue))

                angle_cutoff = 60
                idx = np.where((angle_n < angle_cutoff) & (angle_n > -angle_cutoff))[0][:num_ue]
                angle = angle_n[idx]
            elif param.ue_distribution_azimuth.upper() == "UNIFORM":
                azimuth_range = (-60, 60)
                angle = (azimuth_range[1] - azimuth_range[0]) * random_number_gen.random_sample(num_ue) \
                        + azimuth_range[0]
            else:
                sys.stderr.write("ERROR\nInvalid UE azimuth distribution: " + param.ue_distribution_distance)
                sys.exit(1)

            for ap in range(num_ap):
                idx = [i for i in range(ap * num_ue_per_ap, ap * num_ue_per_ap + num_ue_per_ap)]
                # theta is the horizontal angle of the UE wrt the serving AP
                theta = topology.azimuth[ap] + angle[idx]
                # calculate UE position in x-y coordinates
                x = topology.x[ap] + radius[idx] * np.cos(np.radians(theta))
                y = topology.y[ap] + radius[idx] * np.sin(np.radians(theta))
                ue_x.extend(x)
                ue_y.extend(y)

                # calculate UE azimuth wrt serving AP
                rlan_ue.azimuth[idx] = (azimuth[idx] + theta + 180) % 360

                # calculate elevation angle
                # psi is the vertical angle of the UE wrt the serving AP
                distance = np.sqrt((topology.x[ap] - x) ** 2 + (topology.y[ap] - y) ** 2)
                psi = np.degrees(np.arctan((param.ap_height - param.ue_height) / distance))
                rlan_ue.elevation[idx] = elevation[idx] + psi
        else:
            sys.stderr.write("ERROR\nInvalid UE distribution type: " + param.ue_distribution_type)
            sys.exit(1)

        rlan_ue.x = np.array(ue_x)
        rlan_ue.y = np.array(ue_y)

        rlan_ue.active = np.zeros(num_ue, dtype=bool)
        rlan_ue.height = param.ue_height*np.ones(num_ue)
        rlan_ue.indoor = random_number_gen.random_sample(num_ue) <= (param.ue_indoor_percent/100)
        rlan_ue.rx_interference = -500*np.ones(num_ue)
        rlan_ue.ext_interference = -500*np.ones(num_ue)

        # TODO: this piece of code works only for uplink
        par = param_ant.get_antenna_parameters("UE","TX")
        for i in range(num_ue):
            rlan_ue.antenna[i] = AntennaBeamformingRlan(par, rlan_ue.azimuth[i],
                                                           rlan_ue.elevation[i],param.rlan_type)

        #rlan_ue.antenna = [AntennaOmni(0) for ap in range(num_ue)]
        rlan_ue.bandwidth = param.bandwidth*np.ones(num_ue)
        rlan_ue.center_freq = param.frequency*np.ones(num_ue)
        rlan_ue.noise_figure = param.ue_noise_figure*np.ones(num_ue)

        if param.spectral_mask == "RLAN":
            rlan_ue.spectral_mask = SpectralMaskRlan(StationType.RLAN_UE,param.frequency,\
                                                   param.bandwidth,scenario = "OUTDOOR")

        elif param.spectral_mask == "LAA":
            rlan_ue.spectral_mask = SpectralMaskLaa(StationType.RLAN_UE,param.frequency,\
                                                   param.bandwidth)

        rlan_ue.spectral_mask.set_mask()
        
        if param.topology == 'MACROCELL' or param.topology == 'HOTSPOT':
            rlan_ue.intersite_dist = param.intersite_distance

        return rlan_ue


    @staticmethod
    def generate_rlan_ue_indoor(param: ParametersRlan,
                               param_ant: ParametersAntennaRlan,
                               random_number_gen: np.random.RandomState,
                               topology: Topology) -> StationManager:
        num_ap = topology.num_access_points
        num_ue_per_ap = param.ue_k*param.ue_k_m
        num_ue = num_ap*num_ue_per_ap

        rlan_ue = StationManager(num_ue)
        rlan_ue.station_type = StationType.RLAN_UE
        ue_x = list()
        ue_y = list()
        ue_z = list()

        # initially set all UE's as indoor
        rlan_ue.indoor = np.ones(num_ue, dtype=bool)

        # Calculate UE pointing
        azimuth_range = (-180, 180)
        azimuth = (azimuth_range[1] - azimuth_range[0])*random_number_gen.random_sample(num_ue) + azimuth_range[0]
        # Remove the randomness from azimuth and you will have a perfect pointing
        #azimuth = np.zeros(num_ue)
        elevation_range = (-90, 90)
        elevation = (elevation_range[1] - elevation_range[0])*random_number_gen.random_sample(num_ue) + elevation_range[0]

        delta_x = (topology.b_w/math.sqrt(topology.ue_indoor_percent) - topology.b_w)/2
        delta_y = (topology.b_d/math.sqrt(topology.ue_indoor_percent) - topology.b_d)/2

        for ap in range(num_ap):
            idx = [i for i in range(ap*num_ue_per_ap, ap*num_ue_per_ap + num_ue_per_ap)]
            # Right most cell of first floor
            if ap % topology.num_cells == 0 and ap < topology.total_ap_level:
                x_min = topology.x[ap] - topology.cell_radius - delta_x
                x_max = topology.x[ap] + topology.cell_radius
            # Left most cell of first floor
            elif ap % topology.num_cells == topology.num_cells - 1 and ap < topology.total_ap_level:
                x_min = topology.x[ap] - topology.cell_radius
                x_max = topology.x[ap] + topology.cell_radius + delta_x
            # Center cells and higher floors
            else:
                x_min = topology.x[ap] - topology.cell_radius
                x_max = topology.x[ap] + topology.cell_radius
            
            # First floor
            if ap < topology.total_ap_level:
                y_min = topology.y[ap] - topology.b_d/2 - delta_y
                y_max = topology.y[ap] + topology.b_d/2 + delta_y
            # Higher floors
            else:
                y_min = topology.y[ap] - topology.b_d/2
                y_max = topology.y[ap] + topology.b_d/2
                
            x = (x_max - x_min)*random_number_gen.random_sample(num_ue_per_ap) + x_min
            y = (y_max - y_min)*random_number_gen.random_sample(num_ue_per_ap) + y_min
            z = [topology.height[ap] - topology.b_h + param.ue_height for k in range(num_ue_per_ap)]
            ue_x.extend(x)
            ue_y.extend(y)
            ue_z.extend(z)

            # theta is the horizontal angle of the UE wrt the serving AP
            theta = np.degrees(np.arctan2(y - topology.y[ap], x - topology.x[ap]))
            # calculate UE azimuth wrt serving AP
            rlan_ue.azimuth[idx] = (azimuth[idx] + theta + 180)%360

            # calculate elevation angle
            # psi is the vertical angle of the UE wrt the serving AP
            distance = np.sqrt((topology.x[ap] - x)**2 + (topology.y[ap] - y)**2)
            psi = np.degrees(np.arctan((param.ap_height - param.ue_height)/distance))
            rlan_ue.elevation[idx] = elevation[idx] + psi

            # check if UE is indoor
            if ap % topology.num_cells == 0:
                out = (x < topology.x[ap] - topology.cell_radius) | \
                      (y > topology.y[ap] + topology.b_d/2) | \
                      (y < topology.y[ap] - topology.b_d/2)
            elif ap % topology.num_cells == topology.num_cells - 1:
                out = (x > topology.x[ap] + topology.cell_radius) | \
                      (y > topology.y[ap] + topology.b_d/2) | \
                      (y < topology.y[ap] - topology.b_d/2)
            else:
                out = (y > topology.y[ap] + topology.b_d/2) | \
                      (y < topology.y[ap] - topology.b_d/2)
            rlan_ue.indoor[idx] = ~ out

        rlan_ue.x = np.array(ue_x)
        rlan_ue.y = np.array(ue_y)
        rlan_ue.height = np.array(ue_z)

        rlan_ue.active = np.zeros(num_ue, dtype=bool)
        rlan_ue.rx_interference = -500*np.ones(num_ue)
        rlan_ue.ext_interference = -500*np.ones(num_ue)

        # TODO: this piece of code works only for uplink
        par = param_ant.get_antenna_parameters("UE","TX")
        for i in range(num_ue):
            rlan_ue.antenna[i] = AntennaBeamformingRlan(par, rlan_ue.azimuth[i],
                                                         rlan_ue.elevation[i],param.rlan_type)

        #rlan_ue.antenna = [AntennaOmni(0) for ap in range(num_ue)]
        rlan_ue.bandwidth = param.bandwidth*np.ones(num_ue)
        rlan_ue.center_freq = param.frequency*np.ones(num_ue)
        rlan_ue.noise_figure = param.ue_noise_figure*np.ones(num_ue)

        if param.spectral_mask == "RLAN":
            rlan_ue.spectral_mask = SpectralMaskRlan(StationType.RLAN_UE,param.frequency,\
                                                   param.bandwidth,scenario = "INDOOR")

        elif param.spectral_mask == "LAA":
            rlan_ue.spectral_mask = SpectralMaskLaa(StationType.RLAN_UE,param.frequency,\
                                                   param.bandwidth)

        rlan_ue.spectral_mask.set_mask()

        return rlan_ue


    @staticmethod
    def generate_system(parameters: Parameters, topology: Topology, random_number_gen: np.random.RandomState ):
        if parameters.general.system == "FSS_ES":
            return StationFactory.generate_fss_earth_station(parameters.fss_es, random_number_gen, topology)
        if parameters.general.system == "AMT_GS":
            return StationFactory.generate_amt_ground_station(parameters.amt_gs, random_number_gen, topology)
        if parameters.general.system == "AMAX_BS":
            return StationFactory.generate_aeromax_base_station(parameters.amax_bs, random_number_gen, topology)
        if parameters.general.system == "AMAX_CPE":
            return StationFactory.generate_aeromax_cpe_station(parameters.amax_cpe, random_number_gen, topology)
        if parameters.general.system == "RDR_GS":
            return StationFactory.generate_radar_ground_station(parameters.rdr_gs, random_number_gen, topology)
        elif parameters.general.system == "FSS_SS":
            return StationFactory.generate_fss_space_station(parameters.fss_ss)
        elif parameters.general.system == "FS":
            return StationFactory.generate_fs_station(parameters.fs)
        elif parameters.general.system == "HAPS":
            return StationFactory.generate_haps(parameters.haps, parameters.rlan.intersite_distance, random_number_gen)
        elif parameters.general.system == "RNS":
            return StationFactory.generate_rns(parameters.rns, random_number_gen)
        elif parameters.general.system == "RAS":
            return StationFactory.generate_ras_station(parameters.ras)
        else:
            sys.stderr.write("ERROR\nInvalid system: " + parameters.general.system)
            sys.exit(1)


    @staticmethod
    def generate_fss_space_station(param: ParametersFssSs):
        fss_space_station = StationManager(1)
        fss_space_station.station_type = StationType.FSS_SS

        # now we set the coordinates according to
        # ITU-R P619-1, Attachment A

        # calculate distances to the centre of the Earth
        dist_sat_centre_earth_km = (param.EARTH_RADIUS + param.altitude)/1000
        dist_rlan_centre_earth_km = (param.EARTH_RADIUS + param.rlan_altitude)/1000

        # calculate Cartesian coordinates of satellite, with origin at centre of the Earth
        sat_lat_rad = param.lat_deg * np.pi / 180.
        rlan_long_diff_rad = param.rlan_long_diff_deg * np.pi / 180.
        x1 = dist_sat_centre_earth_km * np.cos(sat_lat_rad) * np.cos(rlan_long_diff_rad)
        y1 = dist_sat_centre_earth_km * np.cos(sat_lat_rad) * np.sin(rlan_long_diff_rad)
        z1 = dist_sat_centre_earth_km * np.sin(sat_lat_rad)

        # rotate axis and calculate coordinates with origin at RLAN system
        rlan_lat_rad = param.rlan_lat_deg * np.pi / 180.
        fss_space_station.x = np.array([x1 * np.sin(rlan_lat_rad) - z1 * np.cos(rlan_lat_rad)]) * 1000
        fss_space_station.y = np.array([y1]) * 1000
        fss_space_station.height = np.array([(z1 * np.sin(rlan_lat_rad) + x1 * np.cos(rlan_lat_rad)
                                             - dist_rlan_centre_earth_km) * 1000])

        fss_space_station.azimuth = param.azimuth
        fss_space_station.elevation = param.elevation

        fss_space_station.active = np.array([True])
        fss_space_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        fss_space_station.rx_interference = -500

        if param.antenna_pattern == "OMNI":
            fss_space_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern == "ITU-R S.672":
            fss_space_station.antenna = np.array([AntennaS672(param)])
        elif param.antenna_pattern == "ITU-R S.1528":
            fss_space_station.antenna = np.array([AntennaS1528(param)])
        elif param.antenna_pattern == "FSS_SS":
            fss_space_station.antenna = np.array([AntennaFssSs(param)])
        else:
            sys.stderr.write("ERROR\nInvalid FSS SS antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        fss_space_station.bandwidth = param.bandwidth
        fss_space_station.noise_temperature = param.noise_temperature
        fss_space_station.thermal_noise = -500
        fss_space_station.total_interference = -500

        return fss_space_station


    @staticmethod

    def generate_fss_earth_station(param: ParametersFssEs, random_number_gen: np.random.RandomState, *args):
        """
        Generates FSS Earth Station.

        Arguments:
            param: ParametersFssEs
            random_number_gen: np.random.RandomState
            topology (optional): Topology
        """
        if len(args): topology = args[0]

        fss_earth_station = StationManager(1)
        fss_earth_station.station_type = StationType.FSS_ES

        if param.location.upper() == "FIXED":
            fss_earth_station.x = np.array([param.x])
            fss_earth_station.y = np.array([param.y])
        elif param.location.upper() == "CELL":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, True)
            fss_earth_station.x = np.array(x)
            fss_earth_station.y = np.array(y)
        elif param.location.upper() == "NETWORK":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, False)
            fss_earth_station.x = np.array(x)
            fss_earth_station.y = np.array(y)
        elif param.location.upper() == "UNIFORM_DIST":
            dist = random_number_gen.uniform( param.min_dist_to_ap, param.max_dist_to_ap)
            angle = random_number_gen.uniform(-np.pi, np.pi)
            fss_earth_station.x[0] = np.array(dist * np.cos(angle))
            fss_earth_station.y[0] = np.array(dist * np.sin(angle))
        else:
            sys.stderr.write("ERROR\nFSS-ES location type {} not supported".format(param.location))
            sys.exit(1)

        fss_earth_station.height = np.array([param.height])

        if param.azimuth.upper() == "RANDOM":
            fss_earth_station.azimuth = random_number_gen.uniform(-180., 180.)
        else:
            fss_earth_station.azimuth = float(param.azimuth)

        elevation = random_number_gen.uniform(param.elevation_min, param.elevation_max)
        fss_earth_station.elevation = np.array([elevation])

        fss_earth_station.active = np.array([True])
        fss_earth_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        fss_earth_station.rx_interference = -500

        if param.antenna_pattern.upper() == "OMNI":
            fss_earth_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern.upper() == "ITU-R S.1855":
            fss_earth_station.antenna = np.array([AntennaS1855(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.465":
            fss_earth_station.antenna = np.array([AntennaS465(param)])
        elif param.antenna_pattern.upper() == "MODIFIED ITU-R S.465":
            fss_earth_station.antenna = np.array([AntennaModifiedS465(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.580":
            fss_earth_station.antenna = np.array([AntennaS580(param)])
        else:
            sys.stderr.write("ERROR\nInvalid FSS ES antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        fss_earth_station.noise_temperature = param.noise_temperature
        fss_earth_station.bandwidth = np.array([param.bandwidth])
        fss_earth_station.noise_temperature = param.noise_temperature
        fss_earth_station.thermal_noise = -500
        fss_earth_station.total_interference = -500

        return fss_earth_station

    @staticmethod

    def generate_amt_ground_station(param: ParametersAmtGs, random_number_gen: np.random.RandomState, *args):
        """
        Generates FSS Earth Station.

        Arguments:
            param: ParametersFssEs
            random_number_gen: np.random.RandomState
            topology (optional): Topology
        """
        if len(args): topology = args[0]

        amt_ground_station = StationManager(1)
        amt_ground_station.station_type = StationType.AMT_GS

        if param.location.upper() == "FIXED":
            amt_ground_station.x = np.array([param.x])
            amt_ground_station.y = np.array([param.y])
        elif param.location.upper() == "CELL":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, True)
            amt_ground_station.x = np.array(x)
            amt_ground_station.y = np.array(y)
        elif param.location.upper() == "NETWORK":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, False)
            amt_ground_station.x = np.array(x)
            amt_ground_station.y = np.array(y)
        elif param.location.upper() == "UNIFORM_DIST":
            dist = random_number_gen.uniform( param.min_dist_to_ap, param.max_dist_to_ap)
            angle = random_number_gen.uniform(-np.pi, np.pi)
            amt_ground_station.x[0] = np.array(dist * np.cos(angle))
            amt_ground_station.y[0] = np.array(dist * np.sin(angle))
        else:
            sys.stderr.write("ERROR\nFSS-ES location type {} not supported".format(param.location))
            sys.exit(1)

        amt_ground_station.height = np.array([param.height])

        if param.azimuth.upper() == "RANDOM":
            amt_ground_station.azimuth = random_number_gen.uniform(-180., 180.)
        else:
            amt_ground_station.azimuth = float(param.azimuth)

        elevation = random_number_gen.uniform(param.elevation_min, param.elevation_max)
        amt_ground_station.elevation = np.array([elevation])

        amt_ground_station.active = np.array([True])
        amt_ground_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        amt_ground_station.rx_interference = -500

        if param.antenna_pattern.upper() == "OMNI":
            amt_ground_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern.upper() == "ITU-R S.1855":
            amt_ground_station.antenna = np.array([AntennaS1855(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.465":
            amt_ground_station.antenna = np.array([AntennaS465(param)])
        elif param.antenna_pattern.upper() == "MODIFIED ITU-R S.465":
            amt_ground_station.antenna = np.array([AntennaModifiedS465(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.580 RLAN":
            amt_ground_station.antenna = np.array([AntennaS580_rlan(param)])
        else:
            sys.stderr.write("ERROR\nInvalid FSS ES antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        amt_ground_station.noise_temperature = param.noise_temperature
        amt_ground_station.bandwidth = np.array([param.bandwidth])
        amt_ground_station.noise_temperature = param.noise_temperature
        amt_ground_station.thermal_noise = -500
        amt_ground_station.total_interference = -500

        return amt_ground_station

    @staticmethod

    def generate_radar_ground_station(param: ParametersRdrGs, random_number_gen: np.random.RandomState, *args):
        """
        Generates FSS Earth Station.

        Arguments:
            param: ParametersFssEs
            random_number_gen: np.random.RandomState
            topology (optional): Topology
        """
        if len(args): topology = args[0]

        radar_ground_station = StationManager(1)
        radar_ground_station.station_type = StationType.RDR_GS

        if param.location.upper() == "FIXED":
            radar_ground_station.x = np.array([param.x])
            radar_ground_station.y = np.array([param.y])
        elif param.location.upper() == "CELL":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, True)
            radar_ground_station.x = np.array(x)
            radar_ground_station.y = np.array(y)
        elif param.location.upper() == "NETWORK":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, False)
            radar_ground_station.x = np.array(x)
            radar_ground_station.y = np.array(y)
        elif param.location.upper() == "UNIFORM_DIST":
            dist = random_number_gen.uniform( param.min_dist_to_ap, param.max_dist_to_ap)
            angle = random_number_gen.uniform(-np.pi, np.pi)
            radar_ground_station.x[0] = np.array(dist * np.cos(angle))
            radar_ground_station.y[0] = np.array(dist * np.sin(angle))
        else:
            sys.stderr.write("ERROR\nFSS-ES location type {} not supported".format(param.location))
            sys.exit(1)

        radar_ground_station.height = np.array([param.height])

        if param.azimuth.upper() == "RANDOM":
            radar_ground_station.azimuth = random_number_gen.uniform(-180., 180.)
        else:
            radar_ground_station.azimuth = float(param.azimuth)

        elevation = random_number_gen.uniform(param.elevation_min, param.elevation_max)
        radar_ground_station.elevation = np.array([elevation])

        radar_ground_station.active = np.array([True])
        radar_ground_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        radar_ground_station.rx_interference = -500

        if param.antenna_pattern.upper() == "OMNI":
            radar_ground_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern.upper() == "ITU-R S.1855":
            radar_ground_station.antenna = np.array([AntennaS1855(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.465":
            radar_ground_station.antenna = np.array([AntennaS465(param)])
        elif param.antenna_pattern.upper() == "MODIFIED ITU-R S.465":
            radar_ground_station.antenna = np.array([AntennaModifiedS465(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.580":
            radar_ground_station.antenna = np.array([AntennaS580(param)])
        elif param.antenna_pattern.upper() == "ITU-R R.1652-1":
            radar_ground_station.antenna = np.array([AntennaRadar(param)])
        else:
            sys.stderr.write("ERROR\nInvalid RDR GS antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        radar_ground_station.noise_temperature = param.noise_temperature
        radar_ground_station.bandwidth = np.array([param.bandwidth])
        radar_ground_station.noise_temperature = param.noise_temperature
        radar_ground_station.thermal_noise = -500
        radar_ground_station.total_interference = -500

        return radar_ground_station

    @staticmethod
    def generate_aeromax_base_station(param: ParametersAmaxBs, random_number_gen: np.random.RandomState, *args):
        """
        Generates FSS Earth Station.

        Arguments:
            param: ParametersFssEs
            random_number_gen: np.random.RandomState
            topology (optional): Topology
        """
        if len(args): topology = args[0]

        aeromax_base_station = StationManager(1)
        aeromax_base_station.station_type = StationType.AMAX_BS

        if param.location.upper() == "FIXED":
            aeromax_base_station.x = np.array([param.x])
            aeromax_base_station.y = np.array([param.y])
        elif param.location.upper() == "CELL":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, True)
            aeromax_base_station.x = np.array(x)
            aeromax_base_station.y = np.array(y)
        elif param.location.upper() == "NETWORK":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, False)
            aeromax_base_station.x = np.array(x)
            aeromax_base_station.y = np.array(y)
        elif param.location.upper() == "UNIFORM_DIST":
            dist = random_number_gen.uniform( param.min_dist_to_ap, param.max_dist_to_ap)
            angle = random_number_gen.uniform(-np.pi, np.pi)
            aeromax_base_station.x[0] = np.array(dist * np.cos(angle))
            aeromax_base_station.y[0] = np.array(dist * np.sin(angle))
        else:
            sys.stderr.write("ERROR\nFSS-ES location type {} not supported".format(param.location))
            sys.exit(1)

        aeromax_base_station.height = np.array([param.height])

        if param.azimuth.upper() == "RANDOM":
            aeromax_base_station.azimuth = random_number_gen.uniform(-180., 180.)
        else:
            aeromax_base_station.azimuth = float(param.azimuth)

        elevation = random_number_gen.uniform(param.elevation_min, param.elevation_max)
        aeromax_base_station.elevation = np.array([elevation])

        aeromax_base_station.active = np.array([True])
        aeromax_base_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        aeromax_base_station.rx_interference = -500

        if param.antenna_pattern.upper() == "OMNI":
            aeromax_base_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern.upper() == "ITU-R S.1855":
            aeromax_base_station.antenna = np.array([AntennaS1855(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.465":
            aeromax_base_station.antenna = np.array([AntennaS465(param)])
        elif param.antenna_pattern.upper() == "MODIFIED ITU-R S.465":
            aeromax_base_station.antenna = np.array([AntennaModifiedS465(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.580":
            aeromax_base_station.antenna = np.array([AntennaS580(param)])
        elif param.antenna_pattern.upper() == "F1336":
            aeromax_base_station.antenna = np.array([AntennaElementAeromaxF1336(param)])
        else:
            sys.stderr.write("ERROR\nInvalid AMAX BS antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        aeromax_base_station.noise_temperature = param.noise_temperature
        aeromax_base_station.bandwidth = np.array([param.bandwidth])
        aeromax_base_station.noise_temperature = param.noise_temperature
        aeromax_base_station.thermal_noise = -500
        aeromax_base_station.total_interference = -500

        return aeromax_base_station

    @staticmethod
    def generate_aeromax_cpe_station(param: ParametersAmaxCpe, random_number_gen: np.random.RandomState, *args):
        """
        Generates FSS Earth Station.

        Arguments:
            param: ParametersFssEs
            random_number_gen: np.random.RandomState
            topology (optional): Topology
        """
        if len(args): topology = args[0]

        aeromax_cpe_station = StationManager(1)
        aeromax_cpe_station.station_type = StationType.AMAX_CPE

        if param.location.upper() == "FIXED":
            aeromax_cpe_station.x = np.array([param.x])
            aeromax_cpe_station.y = np.array([param.y])
        elif param.location.upper() == "CELL":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, True)
            aeromax_cpe_station.x = np.array(x)
            aeromax_cpe_station.y = np.array(y)
        elif param.location.upper() == "NETWORK":
            x, y, dummy1, dummy2 = StationFactory.get_random_position(1, topology, random_number_gen,
                                                                      param.min_dist_to_ap, False)
            aeromax_cpe_station.x = np.array(x)
            aeromax_cpe_station.y = np.array(y)
        elif param.location.upper() == "UNIFORM_DIST":
            dist = random_number_gen.uniform( param.min_dist_to_ap, param.max_dist_to_ap)
            angle = random_number_gen.uniform(-np.pi, np.pi)
            aeromax_cpe_station.x[0] = np.array(dist * np.cos(angle))
            aeromax_cpe_station.y[0] = np.array(dist * np.sin(angle))
        else:
            sys.stderr.write("ERROR\nFSS-ES location type {} not supported".format(param.location))
            sys.exit(1)

        aeromax_cpe_station.height = np.array([param.height])

        if param.azimuth.upper() == "RANDOM":
            aeromax_cpe_station.azimuth = random_number_gen.uniform(-180., 180.)
        else:
            aeromax_cpe_station.azimuth = float(param.azimuth)

        elevation = random_number_gen.uniform(param.elevation_min, param.elevation_max)
        aeromax_cpe_station.elevation = np.array([elevation])

        aeromax_cpe_station.active = np.array([True])
        aeromax_cpe_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        aeromax_cpe_station.rx_interference = -500

        if param.antenna_pattern.upper() == "OMNI":
            aeromax_cpe_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern.upper() == "ITU-R S.1855":
            aeromax_cpe_station.antenna = np.array([AntennaS1855(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.465":
            aeromax_cpe_station.antenna = np.array([AntennaS465(param)])
        elif param.antenna_pattern.upper() == "MODIFIED ITU-R S.465":
            aeromax_cpe_station.antenna = np.array([AntennaModifiedS465(param)])
        elif param.antenna_pattern.upper() == "ITU-R S.580":
            aeromax_cpe_station.antenna = np.array([AntennaS580(param)])
        elif param.antenna_pattern.upper() == "F1336":
            aeromax_cpe_station.antenna = np.array([AntennaElementAeromaxF1336(param)])

        else:
            sys.stderr.write("ERROR\nInvalid AMAX CPE antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        aeromax_cpe_station.noise_temperature = param.noise_temperature
        aeromax_cpe_station.bandwidth = np.array([param.bandwidth])
        aeromax_cpe_station.noise_temperature = param.noise_temperature
        aeromax_cpe_station.thermal_noise = -500
        aeromax_cpe_station.total_interference = -500

        return aeromax_cpe_station


    @staticmethod
    def generate_fs_station(param: ParametersFs):
        fs_station = StationManager(1)
        fs_station.station_type = StationType.FS

        fs_station.x = np.array([param.x])
        fs_station.y = np.array([param.y])
        fs_station.height = np.array([param.height])

        fs_station.azimuth = np.array([param.azimuth])
        fs_station.elevation = np.array([param.elevation])

        fs_station.active = np.array([True])
        fs_station.tx_power = np.array([param.tx_power_density + 10*math.log10(param.bandwidth*1e6) + 30])
        fs_station.rx_interference = -500

        if param.antenna_pattern == "OMNI":
            fs_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern == "ITU-R F.699":
            fs_station.antenna = np.array([AntennaF699(param)])
        else:
            sys.stderr.write("ERROR\nInvalid FS antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        fs_station.noise_temperature = param.noise_temperature
        fs_station.bandwidth = np.array([param.bandwidth])

        return fs_station


    @staticmethod
    def generate_haps(param: ParametersHaps, intersite_distance: int, random_number_gen: np.random.RandomState()):
        num_haps = 1
        haps = StationManager(num_haps)
        haps.station_type = StationType.HAPS

#        d = intersite_distance
#        h = (d/3)*math.sqrt(3)/2
#        haps.x = np.array([0, 7*d/2, -d/2, -4*d, -7*d/2, d/2, 4*d])
#        haps.y = np.array([0, 9*h, 15*h, 6*h, -9*h, -15*h, -6*h])
        haps.x = np.array([0])
        haps.y = np.array([0])

        haps.height = param.altitude * np.ones(num_haps)

        elev_max = 68.19 # corresponds to 50 km radius and 20 km altitude
        haps.azimuth = 360 * random_number_gen.random_sample(num_haps)
        haps.elevation = ((270 + elev_max) - (270 - elev_max)) * random_number_gen.random_sample(num_haps) + \
                         (270 - elev_max)

        haps.active = np.ones(num_haps, dtype = bool)

        haps.antenna = np.empty(num_haps, dtype=Antenna)

        if param.antenna_pattern == "OMNI":
            for i in range(num_haps):
                haps.antenna[i] = AntennaOmni(param.antenna_gain)
        elif param.antenna_pattern == "ITU-R F.1891":
            for i in range(num_haps):
                haps.antenna[i] = AntennaF1891(param)
        else:
            sys.stderr.write("ERROR\nInvalid HAPS (airbone) antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        haps.bandwidth = np.array([param.bandwidth])

        return haps


    @staticmethod
    def generate_rns(param: ParametersRns, random_number_gen: np.random.RandomState()):
        num_rns = 1
        rns = StationManager(num_rns)
        rns.station_type = StationType.RNS

        rns.x = np.array([param.x])
        rns.y = np.array([param.y])
        rns.height = np.array([param.altitude])

        # minimum and maximum values for azimuth and elevation
        azimuth = np.array([-30, 30])
        elevation = np.array([-30, 5])

        rns.azimuth = 90 + (azimuth[1] - azimuth[0]) * random_number_gen.random_sample(num_rns) + azimuth[0]
        rns.elevation = (elevation[1] - elevation[0]) * random_number_gen.random_sample(num_rns) + elevation[0]

        rns.active = np.ones(num_rns, dtype = bool)

        if param.antenna_pattern == "OMNI":
            rns.antenna = np.array([AntennaOmni(param.antenna_gain)])
        elif param.antenna_pattern == "ITU-R M.1466":
            rns.antenna = np.array([AntennaM1466(param.antenna_gain, rns.azimuth, rns.elevation)])
        else:
            sys.stderr.write("ERROR\nInvalid RNS antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        rns.bandwidth = np.array([param.bandwidth])
        rns.noise_temperature = param.noise_temperature
        rns.thermal_noise = -500
        rns.total_interference = -500
        rns.rx_interference = -500

        return rns


    @staticmethod
    def generate_ras_station(param: ParametersRas):
        ras_station = StationManager(1)
        ras_station.station_type = StationType.RAS

        ras_station.x = np.array([param.x])
        ras_station.y = np.array([param.y])
        ras_station.height = np.array([param.height])

        ras_station.azimuth = np.array([param.azimuth])
        ras_station.elevation = np.array([param.elevation])

        ras_station.active = np.array([True])
        ras_station.rx_interference = -500

        if param.antenna_pattern == "OMNI":
            ras_station.antenna = np.array([AntennaOmni(param.antenna_gain)])
            ras_station.antenna[0].effective_area = param.SPEED_OF_LIGHT**2/(4*np.pi*(param.frequency*1e6)**2)
        elif param.antenna_pattern == "ITU-R SA.509":
            ras_station.antenna = np.array([AntennaSA509(param)])
        else:
            sys.stderr.write("ERROR\nInvalid RAS antenna pattern: " + param.antenna_pattern)
            sys.exit(1)

        ras_station.noise_temperature = np.array(param.antenna_noise_temperature + \
                                                  param.receiver_noise_temperature)
        ras_station.bandwidth = np.array(param.bandwidth)

        return ras_station

    @staticmethod
    def get_random_position( num_stas: int, topology: Topology,
                             random_number_gen: np.random.RandomState,
                             min_dist_to_ap = 0, central_cell = False ):
        hexagon_radius = topology.intersite_distance / 3

        min_dist_ok = False

        while not min_dist_ok:
            # generate UE uniformly in a triangle
            x = random_number_gen.uniform(0, hexagon_radius * np.cos(np.pi / 6), num_stas)
            y = random_number_gen.uniform(0, hexagon_radius / 2, num_stas)

            invert_index = np.arctan(y / x) > np.pi / 6
            y[invert_index] = -(hexagon_radius / 2 - y[invert_index])
            x[invert_index] = (hexagon_radius * np.cos(np.pi / 6) - x[invert_index])

            if any (np.sqrt(x**2 + y**2) <  min_dist_to_ap):
                min_dist_ok = False
            else:
                min_dist_ok = True

        # randomly choose an hextant
        hextant = random_number_gen.random_integers(0, 5, num_stas)
        hextant_angle = np.pi / 6 + np.pi / 3 * hextant

        old_x = x
        x = x * np.cos(hextant_angle) - y * np.sin(hextant_angle)
        y = old_x * np.sin(hextant_angle) + y * np.cos(hextant_angle)

        # randomly choose a cell
        if central_cell:
            central_cell_indices = np.where((topology.x == 0) & (topology.y == 0))
            cell = central_cell_indices[0][random_number_gen.random_integers(0, len(central_cell_indices[0]) - 1,
                                                                             num_stas)]
        else:
            num_ap = topology.num_access_points
            cell = random_number_gen.random_integers(0, num_ap - 1, num_stas)

        cell_x = topology.x[cell]
        cell_y = topology.y[cell]

        x = x + cell_x + hexagon_radius * np.cos(topology.azimuth[cell] * np.pi / 180)
        y = y + cell_y + hexagon_radius * np.sin(topology.azimuth[cell] * np.pi / 180)

        x = list(x)
        y = list(y)

        # calculate UE azimuth wrt serving AP
        theta = np.arctan2(y - cell_y, x - cell_x)

        # calculate elevation angle
        # psi is the vertical angle of the UE wrt the serving AP
        distance = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)

        return x, y, theta, distance


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # plot uniform distribution in macrocell scenario

    factory = StationFactory()
    topology = TopologyMacrocell(1000, 1)
    topology.calculate_coordinates()

    class ParamsAux(object):
        def __init__(self):
            self.ue_distribution_type = "UNIFORM"
            self.ap_height = 30
            self.ue_height = 3
            self.ue_indoor_percent = 0
            self.ue_k = 3
            self.ue_k_m = 20
            self.bandwidth  = np.random.rand()
            self.ue_noise_figure = np.random.rand()

    params = ParamsAux()

    ant_param = ParametersAntennaRlan()

    ant_param.ap_element_pattern = "F1336"
    ant_param.ap_tx_element_max_g = 5
    ant_param.ap_tx_element_phi_deg_3db = 65
    ant_param.ap_tx_element_theta_deg_3db = 65
    ant_param.ap_tx_element_am = 30
    ant_param.ap_tx_element_sla_v = 30
    ant_param.ap_tx_n_rows = 8
    ant_param.ap_tx_n_columns = 8
    ant_param.ap_tx_element_horiz_spacing = 0.5
    ant_param.ap_tx_element_vert_spacing = 0.5
    ant_param.ap_downtilt_deg = 10

    ant_param.ue_element_pattern = "FIXED"
    ant_param.ue_tx_element_max_g = 5
    ant_param.ue_tx_element_phi_deg_3db = 90
    ant_param.ue_tx_element_theta_deg_3db = 90
    ant_param.ue_tx_element_am = 25
    ant_param.ue_tx_element_sla_v = 25
    ant_param.ue_tx_n_rows = 4
    ant_param.ue_tx_n_columns = 4
    ant_param.ue_tx_element_horiz_spacing = 0.5
    ant_param.ue_tx_element_vert_spacing = 0.5

    rlan_ue = factory.generate_rlan_ue(params, ant_param, topology)

    fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    topology.plot(ax)

    plt.axis('image')
    plt.title("Macro cell topology")
    plt.xlabel("x-coordinate [m]")
    plt.ylabel("y-coordinate [m]")

    plt.plot(rlan_ue.x, rlan_ue.y, "*")

    plt.tight_layout()
    plt.show()
