# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:35:52 2017

@author: edgar
"""

import configparser

from sharc.parameters.parameters_general import ParametersGeneral
from sharc.parameters.parameters_rlan import ParametersRlan
from sharc.parameters.parameters_hotspot import ParametersHotspot
from sharc.parameters.parameters_indoor import ParametersIndoor
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


class Parameters(object):
    """
    Reads parameters from input file.
    """

    def __init__(self):
        self.file_name = None

        self.general = ParametersGeneral()
        self.rlan = ParametersRlan()
        self.antenna_rlan = ParametersAntennaRlan()
        self.hotspot = ParametersHotspot()
        self.indoor = ParametersIndoor()
        self.fs = ParametersFs()
        self.fss_ss = ParametersFssSs()
        self.fss_es = ParametersFssEs()
        self.amt_gs = ParametersAmtGs()
        self.haps = ParametersHaps()
        self.rns = ParametersRns()
        self.ras = ParametersRas()
        self.rdr_gs = ParametersRdrGs()
        self.amax_bs = ParametersAmaxBs()
        self.amax_cpe = ParametersAmaxCpe()


    def set_file_name(self, file_name: str):
        self.file_name = file_name


    def read_params(self):
        config = configparser.ConfigParser()
        config.read(self.file_name)

        #######################################################################
        # GENERAL
        #######################################################################
        self.general.num_snapshots   = config.getint("GENERAL", "num_snapshots")
        self.general.rlan_link        = config.get("GENERAL", "rlan_link")
        self.general.system          = config.get("GENERAL", "system")
        self.general.enable_cochannel = config.getboolean("GENERAL", "enable_cochannel")
        self.general.enable_adjacent_channel = config.getboolean("GENERAL", "enable_adjacent_channel")
        self.general.seed            = config.get("GENERAL", "seed")
        self.general.overwrite_output = config.getboolean("GENERAL", "overwrite_output")


        #######################################################################
        # RLAN
        #######################################################################
        self.rlan.rlan_type               = config.get("RLAN", "rlan_type")
        self.rlan.topology                = config.get("RLAN", "topology")
        self.rlan.wrap_around             = config.getboolean("RLAN", "wrap_around")
        self.rlan.num_macrocell_sites     = config.getint("RLAN", "num_macrocell_sites")
        self.rlan.num_clusters            = config.getint("RLAN", "num_clusters")
        self.rlan.intersite_distance      = config.getfloat("RLAN", "intersite_distance")
        self.rlan.minimum_separation_distance_ap_ue = config.getfloat("RLAN", "minimum_separation_distance_ap_ue")
        self.rlan.interfered_with         = config.getboolean("RLAN", "interfered_with")
        self.rlan.frequency               = config.getfloat("RLAN", "frequency")
        self.rlan.bandwidth               = config.getfloat("RLAN", "bandwidth")
        self.rlan.rb_bandwidth            = config.getfloat("RLAN", "rb_bandwidth")
        self.rlan.spectral_mask           = config.get("RLAN", "spectral_mask")
        self.rlan.guard_band_ratio        = config.getfloat("RLAN", "guard_band_ratio")
        self.rlan.ap_load_probability     = config.getfloat("RLAN", "ap_load_probability")
        self.rlan.ap_conducted_power      = config.getfloat("RLAN", "ap_conducted_power")
        self.rlan.ap_height               = config.getfloat("RLAN", "ap_height")
        self.rlan.ap_noise_figure         = config.getfloat("RLAN", "ap_noise_figure")
        self.rlan.ap_noise_temperature    = config.getfloat("RLAN", "ap_noise_temperature")
        self.rlan.ap_ohmic_loss           = config.getfloat("RLAN", "ap_ohmic_loss")
        self.rlan.ul_attenuation_factor   = config.getfloat("RLAN", "ul_attenuation_factor")
        self.rlan.ul_sinr_min             = config.getfloat("RLAN", "ul_sinr_min")
        self.rlan.ul_sinr_max             = config.getfloat("RLAN", "ul_sinr_max")
        self.rlan.ue_k                    = config.getint("RLAN", "ue_k")
        self.rlan.ue_k_m                  = config.getint("RLAN", "ue_k_m")
        self.rlan.ue_indoor_percent       = config.getfloat("RLAN", "ue_indoor_percent")
        self.rlan.ue_distribution_type    = config.get("RLAN", "ue_distribution_type")
        self.rlan.ue_distribution_distance = config.get("RLAN", "ue_distribution_distance")
        self.rlan.ue_distribution_azimuth = config.get("RLAN", "ue_distribution_azimuth")
        self.rlan.ue_tx_power_control     = config.get("RLAN", "ue_tx_power_control")
        self.rlan.ue_p_o_pusch            = config.getfloat("RLAN", "ue_p_o_pusch")
        self.rlan.ue_alpha                 = config.getfloat("RLAN", "ue_alpha")
        self.rlan.ue_p_cmax               = config.getfloat("RLAN", "ue_p_cmax")
        self.rlan.ue_height               = config.getfloat("RLAN", "ue_height")
        self.rlan.ue_noise_figure         = config.getfloat("RLAN", "ue_noise_figure")
        self.rlan.ue_ohmic_loss            = config.getfloat("RLAN", "ue_ohmic_loss")
        self.rlan.ue_body_loss            = config.getfloat("RLAN", "ue_body_loss")
        self.rlan.dl_attenuation_factor   = config.getfloat("RLAN", "dl_attenuation_factor")
        self.rlan.dl_sinr_min             = config.getfloat("RLAN", "dl_sinr_min")
        self.rlan.dl_sinr_max             = config.getfloat("RLAN", "dl_sinr_max")
        self.rlan.channel_model           = config.get("RLAN", "channel_model")
        self.rlan.line_of_sight_prob      = config.getfloat("RLAN", "line_of_sight_prob")
        self.rlan.shadowing               = config.getboolean("RLAN", "shadowing")
        self.rlan.noise_temperature       = config.getfloat("RLAN", "noise_temperature")
        self.rlan.BOLTZMANN_CONSTANT      = config.getfloat("RLAN", "BOLTZMANN_CONSTANT")

        #######################################################################
        # RLAN ANTENNA
        #######################################################################
        self.antenna_rlan.normalization              = config.getboolean("RLAN_ANTENNA", "beamforming_normalization")
        self.antenna_rlan.ap_normalization_file          = config.get("RLAN_ANTENNA", "ap_normalization_file")
        self.antenna_rlan.ue_normalization_file          = config.get("RLAN_ANTENNA", "ue_normalization_file")
        self.antenna_rlan.ap_element_pattern          = config.get("RLAN_ANTENNA", "ap_element_pattern")
        self.antenna_rlan.ue_element_pattern          = config.get("RLAN_ANTENNA", "ue_element_pattern")
        self.antenna_rlan.ap_tx_element_max_g         = config.getfloat("RLAN_ANTENNA", "ap_tx_element_max_g")
        self.antenna_rlan.ap_tx_element_phi_deg_3db   = config.getfloat("RLAN_ANTENNA", "ap_tx_element_phi_deg_3db")
        self.antenna_rlan.ap_tx_element_theta_deg_3db = config.getfloat("RLAN_ANTENNA", "ap_tx_element_theta_deg_3db")
        self.antenna_rlan.ap_tx_element_am       = config.getfloat("RLAN_ANTENNA", "ap_tx_element_am")
        self.antenna_rlan.ap_tx_element_sla_v    = config.getfloat("RLAN_ANTENNA", "ap_tx_element_sla_v")
        self.antenna_rlan.ap_tx_n_rows           = config.getfloat("RLAN_ANTENNA", "ap_tx_n_rows")
        self.antenna_rlan.ap_tx_n_columns        = config.getfloat("RLAN_ANTENNA", "ap_tx_n_columns")
        self.antenna_rlan.ap_tx_element_horiz_spacing = config.getfloat("RLAN_ANTENNA", "ap_tx_element_horiz_spacing")
        self.antenna_rlan.ap_tx_element_vert_spacing = config.getfloat("RLAN_ANTENNA", "ap_tx_element_vert_spacing")

        self.antenna_rlan.ap_rx_element_max_g    = config.getfloat("RLAN_ANTENNA", "ap_rx_element_max_g")
        self.antenna_rlan.ap_rx_element_phi_deg_3db  = config.getfloat("RLAN_ANTENNA", "ap_rx_element_phi_deg_3db")
        self.antenna_rlan.ap_rx_element_theta_deg_3db = config.getfloat("RLAN_ANTENNA", "ap_rx_element_theta_deg_3db")
        self.antenna_rlan.ap_rx_element_am       = config.getfloat("RLAN_ANTENNA", "ap_rx_element_am")
        self.antenna_rlan.ap_rx_element_sla_v    = config.getfloat("RLAN_ANTENNA", "ap_rx_element_sla_v")
        self.antenna_rlan.ap_rx_n_rows           = config.getfloat("RLAN_ANTENNA", "ap_rx_n_rows")
        self.antenna_rlan.ap_rx_n_columns        = config.getfloat("RLAN_ANTENNA", "ap_rx_n_columns")
        self.antenna_rlan.ap_rx_element_horiz_spacing = config.getfloat("RLAN_ANTENNA", "ap_rx_element_horiz_spacing")
        self.antenna_rlan.ap_rx_element_vert_spacing = config.getfloat("RLAN_ANTENNA", "ap_rx_element_vert_spacing")

        self.antenna_rlan.ue_tx_element_max_g    = config.getfloat("RLAN_ANTENNA", "ue_tx_element_max_g")
        self.antenna_rlan.ue_tx_element_phi_deg_3db  = config.getfloat("RLAN_ANTENNA", "ue_tx_element_phi_deg_3db")
        self.antenna_rlan.ue_tx_element_theta_deg_3db = config.getfloat("RLAN_ANTENNA", "ue_tx_element_theta_deg_3db")
        self.antenna_rlan.ue_tx_element_am       = config.getfloat("RLAN_ANTENNA", "ue_tx_element_am")
        self.antenna_rlan.ue_tx_element_sla_v    = config.getfloat("RLAN_ANTENNA", "ue_tx_element_sla_v")
        self.antenna_rlan.ue_tx_n_rows           = config.getfloat("RLAN_ANTENNA", "ue_tx_n_rows")
        self.antenna_rlan.ue_tx_n_columns        = config.getfloat("RLAN_ANTENNA", "ue_tx_n_columns")
        self.antenna_rlan.ue_tx_element_horiz_spacing = config.getfloat("RLAN_ANTENNA", "ue_tx_element_horiz_spacing")
        self.antenna_rlan.ue_tx_element_vert_spacing = config.getfloat("RLAN_ANTENNA", "ue_tx_element_vert_spacing")

        self.antenna_rlan.ue_rx_element_max_g    = config.getfloat("RLAN_ANTENNA", "ue_rx_element_max_g")
        self.antenna_rlan.ue_rx_element_phi_deg_3db  = config.getfloat("RLAN_ANTENNA", "ue_rx_element_phi_deg_3db")
        self.antenna_rlan.ue_rx_element_theta_deg_3db = config.getfloat("RLAN_ANTENNA", "ue_rx_element_theta_deg_3db")
        self.antenna_rlan.ue_rx_element_am       = config.getfloat("RLAN_ANTENNA", "ue_rx_element_am")
        self.antenna_rlan.ue_rx_element_sla_v    = config.getfloat("RLAN_ANTENNA", "ue_rx_element_sla_v")
        self.antenna_rlan.ue_rx_n_rows           = config.getfloat("RLAN_ANTENNA", "ue_rx_n_rows")
        self.antenna_rlan.ue_rx_n_columns        = config.getfloat("RLAN_ANTENNA", "ue_rx_n_columns")
        self.antenna_rlan.ue_rx_element_horiz_spacing = config.getfloat("RLAN_ANTENNA", "ue_rx_element_horiz_spacing")
        self.antenna_rlan.ue_rx_element_vert_spacing = config.getfloat("RLAN_ANTENNA", "ue_rx_element_vert_spacing")

        self.antenna_rlan.ap_downtilt_deg = config.getfloat("RLAN_ANTENNA", "ap_downtilt_deg")

        #######################################################################
        # HOTSPOT
        #######################################################################
        self.hotspot.num_hotspots_per_cell = config.getint("HOTSPOT", "num_hotspots_per_cell")
        self.hotspot.max_dist_hotspot_ue   = config.getfloat("HOTSPOT", "max_dist_hotspot_ue")
        self.hotspot.min_dist_ap_hotspot   = config.getfloat("HOTSPOT", "min_dist_ap_hotspot")
        self.hotspot.min_dist_hotspots     = config.getfloat("HOTSPOT", "min_dist_hotspots")

        #######################################################################
        # INDOOR
        #######################################################################
        self.indoor.basic_path_loss = config.get("INDOOR", "basic_path_loss")
        self.indoor.n_rows = config.getint("INDOOR", "n_rows")
        self.indoor.n_colums = config.getint("INDOOR", "n_colums")
        self.indoor.num_rlan_buildings = config.get("INDOOR", "num_rlan_buildings")
        self.indoor.street_width = config.getint("INDOOR", "street_width")
        self.indoor.intersite_distance = config.getfloat("INDOOR", "intersite_distance")
        self.indoor.num_cells = config.getint("INDOOR", "num_cells")
        self.indoor.num_floors = config.getint("INDOOR", "num_floors")
        self.indoor.ue_indoor_percent = config.getfloat("INDOOR", "ue_indoor_percent")
        self.indoor.building_class = config.get("INDOOR", "building_class")

        #######################################################################
        # FSS space station
        #######################################################################
        self.fss_ss.frequency               = config.getfloat("FSS_SS", "frequency")
        self.fss_ss.bandwidth               = config.getfloat("FSS_SS", "bandwidth")
        self.fss_ss.tx_power_density        = config.getfloat("FSS_SS", "tx_power_density")
        self.fss_ss.altitude                = config.getfloat("FSS_SS", "altitude")
        self.fss_ss.lat_deg                 = config.getfloat("FSS_SS", "lat_deg")
        self.fss_ss.elevation               = config.getfloat("FSS_SS", "elevation")
        self.fss_ss.azimuth                 = config.getfloat("FSS_SS", "azimuth")
        self.fss_ss.noise_temperature       = config.getfloat("FSS_SS", "noise_temperature")
        self.fss_ss.adjacent_ch_selectivity = config.getfloat("FSS_SS", "adjacent_ch_selectivity")
        self.fss_ss.inr_scaling             = config.getfloat("FSS_SS", "inr_scaling")
        self.fss_ss.antenna_gain            = config.getfloat("FSS_SS", "antenna_gain")
        self.fss_ss.antenna_pattern         = config.get("FSS_SS", "antenna_pattern")
        self.fss_ss.rlan_altitude            = config.getfloat("FSS_SS", "rlan_altitude")
        self.fss_ss.rlan_lat_deg             = config.getfloat("FSS_SS", "rlan_lat_deg")
        self.fss_ss.rlan_long_diff_deg       = config.getfloat("FSS_SS", "rlan_long_diff_deg")
        self.fss_ss.season                  = config.get("FSS_SS", "season")
        self.fss_ss.channel_model           = config.get("FSS_SS", "channel_model")
        self.fss_ss.antenna_l_s             = config.getfloat("FSS_SS", "antenna_l_s")
        self.fss_ss.antenna_3_dB            = config.getfloat("FSS_SS", "antenna_3_dB")
        self.fss_ss.BOLTZMANN_CONSTANT      = config.getfloat("FSS_SS", "BOLTZMANN_CONSTANT")
        self.fss_ss.EARTH_RADIUS            = config.getfloat("FSS_SS", "EARTH_RADIUS")

        #######################################################################
        # FSS earth station
        #######################################################################
        self.fss_es.location = config.get("FSS_ES", "location")
        self.fss_es.x = config.getfloat("FSS_ES", "x")
        self.fss_es.y = config.getfloat("FSS_ES", "y")
        self.fss_es.min_dist_to_ap = config.getfloat("FSS_ES", "min_dist_to_ap")
        self.fss_es.max_dist_to_ap = config.getfloat("FSS_ES", "max_dist_to_ap")
        self.fss_es.height = config.getfloat("FSS_ES", "height")
        self.fss_es.elevation_min = config.getfloat("FSS_ES", "elevation_min")
        self.fss_es.elevation_max = config.getfloat("FSS_ES", "elevation_max")
        self.fss_es.azimuth = config.get("FSS_ES", "azimuth")
        self.fss_es.frequency = config.getfloat("FSS_ES", "frequency")
        self.fss_es.bandwidth = config.getfloat("FSS_ES", "bandwidth")
        self.fss_es.adjacent_ch_selectivity = config.getfloat("FSS_ES", "adjacent_ch_selectivity")
        self.fss_es.tx_power_density = config.getfloat("FSS_ES", "tx_power_density")
        self.fss_es.noise_temperature = config.getfloat("FSS_ES", "noise_temperature")
        self.fss_es.inr_scaling = config.getfloat("FSS_ES", "inr_scaling")
        self.fss_es.antenna_gain = config.getfloat("FSS_ES", "antenna_gain")
        self.fss_es.antenna_pattern = config.get("FSS_ES", "antenna_pattern")
        self.fss_es.antenna_envelope_gain = config.getfloat("FSS_ES", "antenna_envelope_gain")
        self.fss_es.diameter = config.getfloat("FSS_ES", "diameter")
        self.fss_es.channel_model = config.get("FSS_ES", "channel_model")
        self.fss_es.line_of_sight_prob = config.getfloat("FSS_ES", "line_of_sight_prob")
        self.fss_es.BOLTZMANN_CONSTANT = config.getfloat("FSS_ES", "BOLTZMANN_CONSTANT")
        self.fss_es.EARTH_RADIUS = config.getfloat("FSS_ES", "EARTH_RADIUS")

        # P452 parameters
        self.fss_es.atmospheric_pressure = config.getfloat("FSS_ES", "atmospheric_pressure")
        self.fss_es.air_temperature = config.getfloat("FSS_ES", "air_temperature")
        self.fss_es.N0 = config.getfloat("FSS_ES", "N0")
        self.fss_es.delta_N = config.getfloat("FSS_ES", "delta_N")
        self.fss_es.percentage_p = config.get("FSS_ES", "percentage_p")
        self.fss_es.Dct = config.getfloat("FSS_ES", "Dct")
        self.fss_es.Dcr = config.getfloat("FSS_ES", "Dcr")
        self.fss_es.Hte = config.getfloat("FSS_ES", "Hte")
        self.fss_es.Hre = config.getfloat("FSS_ES", "Hre")
        self.fss_es.tx_lat = config.getfloat("FSS_ES", "tx_lat")
        self.fss_es.rx_lat = config.getfloat("FSS_ES", "rx_lat")
        self.fss_es.polarization = config.get("FSS_ES", "polarization")
        self.fss_es.clutter_loss = config.getboolean("FSS_ES", "clutter_loss")
        
        # HDFSS propagation parameters
        self.fss_es.es_position = config.get("FSS_ES", "es_position")
        self.fss_es.shadow_enabled = config.getboolean("FSS_ES", "shadow_enabled")
        self.fss_es.building_loss_enabled = config.getboolean("FSS_ES", "building_loss_enabled")
        self.fss_es.same_building_enabled = config.getboolean("FSS_ES", "same_building_enabled")
        self.fss_es.diffraction_enabled = config.getboolean("FSS_ES", "diffraction_enabled")
        self.fss_es.ap_building_entry_loss_type = config.get("FSS_ES", "ap_building_entry_loss_type")
        self.fss_es.ap_building_entry_loss_prob = config.getfloat("FSS_ES", "ap_building_entry_loss_prob")
        self.fss_es.ap_building_entry_loss_value = config.getfloat("FSS_ES", "ap_building_entry_loss_value")

        #######################################################################
        # Fixed wireless service
        #######################################################################
        self.fs.x                       = config.getfloat("FS", "x")
        self.fs.y                       = config.getfloat("FS", "y")
        self.fs.height                  = config.getfloat("FS", "height")
        self.fs.elevation               = config.getfloat("FS", "elevation")
        self.fs.azimuth                 = config.getfloat("FS", "azimuth")
        self.fs.frequency               = config.getfloat("FS", "frequency")
        self.fs.bandwidth               = config.getfloat("FS", "bandwidth")
        self.fs.noise_temperature       = config.getfloat("FS", "noise_temperature")
        self.fs.adjacent_ch_selectivity = config.getfloat("FS", "adjacent_ch_selectivity")
        self.fs.tx_power_density        = config.getfloat("FS", "tx_power_density")
        self.fs.inr_scaling             = config.getfloat("FS", "inr_scaling")
        self.fs.antenna_gain            = config.getfloat("FS", "antenna_gain")
        self.fs.antenna_pattern         = config.get("FS", "antenna_pattern")
        self.fs.diameter                = config.getfloat("FS", "diameter")
        self.fs.channel_model           = config.get("FS", "channel_model")
        self.fs.line_of_sight_prob      = config.getfloat("FS", "line_of_sight_prob")
        self.fs.BOLTZMANN_CONSTANT      = config.getfloat("FS", "BOLTZMANN_CONSTANT")
        self.fs.EARTH_RADIUS            = config.getfloat("FS", "EARTH_RADIUS")

        #######################################################################
        # HAPS (airbone) station
        #######################################################################
        self.haps.frequency               = config.getfloat("HAPS", "frequency")
        self.haps.bandwidth               = config.getfloat("HAPS", "bandwidth")
        self.haps.antenna_gain            = config.getfloat("HAPS", "antenna_gain")
        self.haps.tx_power_density        = config.getfloat("HAPS", "eirp_density") - self.haps.antenna_gain - 60
        self.haps.altitude                = config.getfloat("HAPS", "altitude")
        self.haps.lat_deg                 = config.getfloat("HAPS", "lat_deg")
        self.haps.elevation               = config.getfloat("HAPS", "elevation")
        self.haps.azimuth                 = config.getfloat("HAPS", "azimuth")
        self.haps.inr_scaling             = config.getfloat("HAPS", "inr_scaling")
        self.haps.antenna_pattern         = config.get("HAPS", "antenna_pattern")
        self.haps.rlan_altitude            = config.getfloat("HAPS", "rlan_altitude")
        self.haps.rlan_lat_deg             = config.getfloat("HAPS", "rlan_lat_deg")
        self.haps.rlan_long_diff_deg       = config.getfloat("HAPS", "rlan_long_diff_deg")
        self.haps.season                  = config.get("HAPS", "season")
        self.haps.acs                     = config.getfloat("HAPS", "acs")
        self.haps.channel_model           = config.get("HAPS", "channel_model")
        self.haps.antenna_l_n             = config.getfloat("HAPS", "antenna_l_n")
        self.haps.BOLTZMANN_CONSTANT      = config.getfloat("HAPS", "BOLTZMANN_CONSTANT")
        self.haps.EARTH_RADIUS            = config.getfloat("HAPS", "EARTH_RADIUS")

        #######################################################################
        # RNS
        #######################################################################
        self.rns.x                  = config.getfloat("RNS", "x")
        self.rns.y                  = config.getfloat("RNS", "y")
        self.rns.altitude           = config.getfloat("RNS", "altitude")
        self.rns.frequency          = config.getfloat("RNS", "frequency")
        self.rns.bandwidth          = config.getfloat("RNS", "bandwidth")
        self.rns.noise_temperature  = config.getfloat("RNS", "noise_temperature")
        self.rns.inr_scaling        = config.getfloat("RNS", "inr_scaling")
        self.rns.tx_power_density   = config.getfloat("RNS", "tx_power_density")
        self.rns.antenna_gain       = config.getfloat("RNS", "antenna_gain")
        self.rns.antenna_pattern    = config.get("RNS", "antenna_pattern")
        self.rns.season             = config.get("RNS", "season")
        self.rns.rlan_altitude       = config.getfloat("RNS", "rlan_altitude")
        self.rns.rlan_lat_deg        = config.getfloat("RNS", "rlan_lat_deg")
        self.rns.channel_model      = config.get("RNS", "channel_model")
        self.rns.acs                = config.getfloat("RNS", "acs")
        self.rns.BOLTZMANN_CONSTANT = config.getfloat("RNS", "BOLTZMANN_CONSTANT")
        self.rns.EARTH_RADIUS       = config.getfloat("RNS", "EARTH_RADIUS")

        #######################################################################
        # RAS station
        #######################################################################
        self.ras.x                          = config.getfloat("RAS", "x")
        self.ras.y                          = config.getfloat("RAS", "y")
        self.ras.height                     = config.getfloat("RAS", "height")
        self.ras.elevation                  = config.getfloat("RAS", "elevation")
        self.ras.azimuth                    = config.getfloat("RAS", "azimuth")
        self.ras.frequency                  = config.getfloat("RAS", "frequency")
        self.ras.bandwidth                  = config.getfloat("RAS", "bandwidth")
        self.ras.antenna_noise_temperature  = config.getfloat("RAS", "antenna_noise_temperature")
        self.ras.receiver_noise_temperature = config.getfloat("RAS", "receiver_noise_temperature")
        self.ras.adjacent_ch_selectivity    = config.getfloat("FSS_ES", "adjacent_ch_selectivity")
        self.ras.inr_scaling                = config.getfloat("RAS", "inr_scaling")
        self.ras.antenna_efficiency         = config.getfloat("RAS", "antenna_efficiency")
        self.ras.antenna_gain               = config.getfloat("RAS", "antenna_gain")
        self.ras.antenna_pattern            = config.get("RAS", "antenna_pattern")
        self.ras.diameter                   = config.getfloat("RAS", "diameter")
        self.ras.channel_model              = config.get("RAS", "channel_model")
        self.ras.line_of_sight_prob         = config.getfloat("RAS", "line_of_sight_prob")
        self.ras.BOLTZMANN_CONSTANT         = config.getfloat("RAS", "BOLTZMANN_CONSTANT")
        self.ras.EARTH_RADIUS               = config.getfloat("RAS", "EARTH_RADIUS")
        self.ras.SPEED_OF_LIGHT             = config.getfloat("RAS", "SPEED_OF_LIGHT")

        # P452 parameters
        self.ras.atmospheric_pressure = config.getfloat("RAS", "atmospheric_pressure")
        self.ras.air_temperature = config.getfloat("RAS", "air_temperature")
        self.ras.N0 = config.getfloat("RAS", "N0")
        self.ras.delta_N = config.getfloat("RAS", "delta_N")
        self.ras.percentage_p = config.get("RAS", "percentage_p")
        self.ras.Dct = config.getfloat("RAS", "Dct")
        self.ras.Dcr = config.getfloat("RAS", "Dcr")
        self.ras.Hte = config.getfloat("RAS", "Hte")
        self.ras.Hre = config.getfloat("RAS", "Hre")
        self.ras.tx_lat = config.getfloat("RAS", "tx_lat")
        self.ras.rx_lat = config.getfloat("RAS", "rx_lat")
        self.ras.polarization = config.get("RAS", "polarization")
        self.ras.clutter_loss = config.getboolean("RAS", "clutter_loss")

        #######################################################################
        # AMT Ground station
        #######################################################################
        self.amt_gs.location = config.get("AMT_GS", "location")
        self.amt_gs.x = config.getfloat("AMT_GS", "x")
        self.amt_gs.y = config.getfloat("AMT_GS", "y")
        self.amt_gs.min_dist_to_ap = config.getfloat("AMT_GS", "min_dist_to_ap")
        self.amt_gs.max_dist_to_ap = config.getfloat("AMT_GS", "max_dist_to_ap")
        self.amt_gs.height = config.getfloat("AMT_GS", "height")
        self.amt_gs.elevation_min = config.getfloat("AMT_GS", "elevation_min")
        self.amt_gs.elevation_max = config.getfloat("AMT_GS", "elevation_max")
        self.amt_gs.azimuth = config.get("AMT_GS", "azimuth")
        self.amt_gs.frequency = config.getfloat("AMT_GS", "frequency")
        self.amt_gs.bandwidth = config.getfloat("AMT_GS", "bandwidth")
        self.amt_gs.adjacent_ch_selectivity = config.getfloat("AMT_GS", "adjacent_ch_selectivity")
        self.amt_gs.tx_power_density = config.getfloat("AMT_GS", "tx_power_density")
        self.amt_gs.noise_temperature = config.getfloat("AMT_GS", "noise_temperature")
        self.amt_gs.inr_scaling = config.getfloat("AMT_GS", "inr_scaling")
        self.amt_gs.antenna_gain = config.getfloat("AMT_GS", "antenna_gain")
        self.amt_gs.antenna_pattern = config.get("AMT_GS", "antenna_pattern")
        self.amt_gs.antenna_envelope_gain = config.getfloat("AMT_GS", "antenna_envelope_gain")
        self.amt_gs.diameter = config.getfloat("AMT_GS", "diameter")
        self.amt_gs.channel_model = config.get("AMT_GS", "channel_model")
        self.amt_gs.line_of_sight_prob = config.getfloat("AMT_GS", "line_of_sight_prob")
        self.amt_gs.BOLTZMANN_CONSTANT = config.getfloat("AMT_GS", "BOLTZMANN_CONSTANT")
        self.amt_gs.EARTH_RADIUS = config.getfloat("AMT_GS", "EARTH_RADIUS")

        # P452 parameters
        self.amt_gs.atmospheric_pressure = config.getfloat("AMT_GS", "atmospheric_pressure")
        self.amt_gs.air_temperature = config.getfloat("AMT_GS", "air_temperature")
        self.amt_gs.N0 = config.getfloat("AMT_GS", "N0")
        self.amt_gs.delta_N = config.getfloat("AMT_GS", "delta_N")
        self.amt_gs.percentage_p = config.get("AMT_GS", "percentage_p")
        self.amt_gs.Dct = config.getfloat("AMT_GS", "Dct")
        self.amt_gs.Dcr = config.getfloat("AMT_GS", "Dcr")
        self.amt_gs.Hte = config.getfloat("AMT_GS", "Hte")
        self.amt_gs.Hre = config.getfloat("AMT_GS", "Hre")
        self.amt_gs.tx_lat = config.getfloat("AMT_GS", "tx_lat")
        self.amt_gs.rx_lat = config.getfloat("AMT_GS", "rx_lat")
        self.amt_gs.polarization = config.get("AMT_GS", "polarization")
        self.amt_gs.clutter_loss = config.getboolean("AMT_GS", "clutter_loss")
        
        # RADAR for indoor interference propagation parameters
        self.amt_gs.es_position = config.get("AMT_GS", "es_position")
        self.amt_gs.shadow_enabled = config.getboolean("AMT_GS", "shadow_enabled")
        self.amt_gs.building_loss_enabled = config.getboolean("AMT_GS", "building_loss_enabled")
        self.amt_gs.same_building_enabled = config.getboolean("AMT_GS", "same_building_enabled")
        self.amt_gs.diffraction_enabled = config.getboolean("AMT_GS", "diffraction_enabled")
        self.amt_gs.ap_building_entry_loss_type = config.get("AMT_GS", "ap_building_entry_loss_type")
        self.amt_gs.ap_building_entry_loss_prob = config.getfloat("AMT_GS", "ap_building_entry_loss_prob")
        self.amt_gs.ap_building_entry_loss_value = config.getfloat("AMT_GS", "ap_building_entry_loss_value")

        #######################################################################
        # RADAR Ground station
        #######################################################################
        self.rdr_gs.location = config.get("RDR_GS", "location")
        self.rdr_gs.x = config.getfloat("RDR_GS", "x")
        self.rdr_gs.y = config.getfloat("RDR_GS", "y")
        self.rdr_gs.min_dist_to_ap = config.getfloat("RDR_GS", "min_dist_to_ap")
        self.rdr_gs.max_dist_to_ap = config.getfloat("RDR_GS", "max_dist_to_ap")
        self.rdr_gs.height = config.getfloat("RDR_GS", "height")
        self.rdr_gs.elevation_min = config.getfloat("RDR_GS", "elevation_min")
        self.rdr_gs.elevation_max = config.getfloat("RDR_GS", "elevation_max")
        self.rdr_gs.azimuth = config.get("RDR_GS", "azimuth")
        self.rdr_gs.frequency = config.getfloat("RDR_GS", "frequency")
        self.rdr_gs.bandwidth = config.getfloat("RDR_GS", "bandwidth")
        self.rdr_gs.adjacent_ch_selectivity = config.getfloat("RDR_GS", "adjacent_ch_selectivity")
        self.rdr_gs.tx_power_density = config.getfloat("RDR_GS", "tx_power_density")
        self.rdr_gs.noise_temperature = config.getfloat("RDR_GS", "noise_temperature")
        self.rdr_gs.inr_scaling = config.getfloat("RDR_GS", "inr_scaling")
        self.rdr_gs.antenna_gain = config.getfloat("RDR_GS", "antenna_gain")
        self.rdr_gs.antenna_pattern = config.get("RDR_GS", "antenna_pattern")
        self.rdr_gs.antenna_envelope_gain = config.getfloat("RDR_GS", "antenna_envelope_gain")
        self.rdr_gs.diameter = config.getfloat("RDR_GS", "diameter")
        self.rdr_gs.channel_model = config.get("RDR_GS", "channel_model")
        self.rdr_gs.line_of_sight_prob = config.getfloat("RDR_GS", "line_of_sight_prob")
        self.rdr_gs.BOLTZMANN_CONSTANT = config.getfloat("RDR_GS", "BOLTZMANN_CONSTANT")
        self.rdr_gs.EARTH_RADIUS = config.getfloat("RDR_GS", "EARTH_RADIUS")

        # P452 parameters
        self.rdr_gs.atmospheric_pressure = config.getfloat("RDR_GS", "atmospheric_pressure")
        self.rdr_gs.air_temperature = config.getfloat("RDR_GS", "air_temperature")
        self.rdr_gs.N0 = config.getfloat("RDR_GS", "N0")
        self.rdr_gs.delta_N = config.getfloat("RDR_GS", "delta_N")
        self.rdr_gs.percentage_p = config.get("RDR_GS", "percentage_p")
        self.rdr_gs.Dct = config.getfloat("RDR_GS", "Dct")
        self.rdr_gs.Dcr = config.getfloat("RDR_GS", "Dcr")
        self.rdr_gs.Hte = config.getfloat("RDR_GS", "Hte")
        self.rdr_gs.Hre = config.getfloat("RDR_GS", "Hre")
        self.rdr_gs.tx_lat = config.getfloat("RDR_GS", "tx_lat")
        self.rdr_gs.rx_lat = config.getfloat("RDR_GS", "rx_lat")
        self.rdr_gs.polarization = config.get("RDR_GS", "polarization")
        self.rdr_gs.clutter_loss = config.getboolean("RDR_GS", "clutter_loss")
        
        # Radar for indoor interference propagation parameters
        self.rdr_gs.es_position = config.get("RDR_GS", "es_position")
        self.rdr_gs.shadow_enabled = config.getboolean("RDR_GS", "shadow_enabled")
        self.rdr_gs.building_loss_enabled = config.getboolean("RDR_GS", "building_loss_enabled")
        self.rdr_gs.same_building_enabled = config.getboolean("RDR_GS", "same_building_enabled")
        self.rdr_gs.diffraction_enabled = config.getboolean("RDR_GS", "diffraction_enabled")
        self.rdr_gs.ap_building_entry_loss_type = config.get("RDR_GS", "ap_building_entry_loss_type")
        self.rdr_gs.ap_building_entry_loss_prob = config.getfloat("RDR_GS", "ap_building_entry_loss_prob")
        self.rdr_gs.ap_building_entry_loss_value = config.getfloat("RDR_GS", "ap_building_entry_loss_value")

        #######################################################################
        # AEROMAX Base station
        #######################################################################
        self.amax_bs.location = config.get("AMAX_BS", "location")
        self.amax_bs.x = config.getfloat("AMAX_BS", "x")
        self.amax_bs.y = config.getfloat("AMAX_BS", "y")
        self.amax_bs.min_dist_to_ap = config.getfloat("AMAX_BS", "min_dist_to_ap")
        self.amax_bs.max_dist_to_ap = config.getfloat("AMAX_BS", "max_dist_to_ap")
        self.amax_bs.height = config.getfloat("AMAX_BS", "height")
        self.amax_bs.elevation_min = config.getfloat("AMAX_BS", "elevation_min")
        self.amax_bs.elevation_max = config.getfloat("AMAX_BS", "elevation_max")
        self.amax_bs.azimuth = config.get("AMAX_BS", "azimuth")
        self.amax_bs.frequency = config.getfloat("AMAX_BS", "frequency")
        self.amax_bs.bandwidth = config.getfloat("AMAX_BS", "bandwidth")
        self.amax_bs.adjacent_ch_selectivity = config.getfloat("AMAX_BS", "adjacent_ch_selectivity")
        self.amax_bs.tx_power_density = config.getfloat("AMAX_BS", "tx_power_density")
        self.amax_bs.noise_temperature = config.getfloat("AMAX_BS", "noise_temperature")
        self.amax_bs.inr_scaling = config.getfloat("AMAX_BS", "inr_scaling")
        self.amax_bs.antenna_gain = config.getfloat("AMAX_BS", "antenna_gain")
        self.amax_bs.antenna_pattern = config.get("AMAX_BS", "antenna_pattern")
        self.amax_bs.antenna_envelope_gain = config.getfloat("AMAX_BS", "antenna_envelope_gain")
        self.amax_bs.diameter = config.getfloat("AMAX_BS", "diameter")
        self.amax_bs.channel_model = config.get("AMAX_BS", "channel_model")
        self.amax_bs.line_of_sight_prob = config.getfloat("AMAX_BS", "line_of_sight_prob")
        self.amax_bs.BOLTZMANN_CONSTANT = config.getfloat("AMAX_BS", "BOLTZMANN_CONSTANT")
        self.amax_bs.EARTH_RADIUS = config.getfloat("AMAX_BS", "EARTH_RADIUS")
        self.amax_bs.downtilt_deg = config.getfloat("AMAX_BS", "downtilt_deg")
        self.amax_bs.element_phi_deg_3db  = config.getfloat("AMAX_BS", "amax_phi_deg_3db")
        self.amax_bs.element_theta_deg_3db = config.getfloat("AMAX_BS", "amax_theta_deg_3db")

        # P452 parameters
        self.amax_bs.atmospheric_pressure = config.getfloat("AMAX_BS", "atmospheric_pressure")
        self.amax_bs.air_temperature = config.getfloat("AMAX_BS", "air_temperature")
        self.amax_bs.N0 = config.getfloat("AMAX_BS", "N0")
        self.amax_bs.delta_N = config.getfloat("AMAX_BS", "delta_N")
        self.amax_bs.percentage_p = config.get("AMAX_BS", "percentage_p")
        self.amax_bs.Dct = config.getfloat("AMAX_BS", "Dct")
        self.amax_bs.Dcr = config.getfloat("AMAX_BS", "Dcr")
        self.amax_bs.Hte = config.getfloat("AMAX_BS", "Hte")
        self.amax_bs.Hre = config.getfloat("AMAX_BS", "Hre")
        self.amax_bs.tx_lat = config.getfloat("AMAX_BS", "tx_lat")
        self.amax_bs.rx_lat = config.getfloat("AMAX_BS", "rx_lat")
        self.amax_bs.polarization = config.get("AMAX_BS", "polarization")
        self.amax_bs.clutter_loss = config.getboolean("AMAX_BS", "clutter_loss")
        
        # Radar for indoor interference propagation parameters
        self.amax_bs.es_position = config.get("AMAX_BS", "es_position")
        self.amax_bs.shadow_enabled = config.getboolean("AMAX_BS", "shadow_enabled")
        self.amax_bs.building_loss_enabled = config.getboolean("AMAX_BS", "building_loss_enabled")
        self.amax_bs.same_building_enabled = config.getboolean("AMAX_BS", "same_building_enabled")
        self.amax_bs.diffraction_enabled = config.getboolean("AMAX_BS", "diffraction_enabled")
        self.amax_bs.ap_building_entry_loss_type = config.get("AMAX_BS", "ap_building_entry_loss_type")
        self.amax_bs.ap_building_entry_loss_prob = config.getfloat("AMAX_BS", "ap_building_entry_loss_prob")
        self.amax_bs.ap_building_entry_loss_value = config.getfloat("AMAX_BS", "ap_building_entry_loss_value")

        #######################################################################
        # AEROMAX CPE station
        #######################################################################
        self.amax_cpe.location = config.get("AMAX_CPE", "location")
        self.amax_cpe.x = config.getfloat("AMAX_CPE", "x")
        self.amax_cpe.y = config.getfloat("AMAX_CPE", "y")
        self.amax_cpe.min_dist_to_ap = config.getfloat("AMAX_CPE", "min_dist_to_ap")
        self.amax_cpe.max_dist_to_ap = config.getfloat("AMAX_CPE", "max_dist_to_ap")
        self.amax_cpe.height = config.getfloat("AMAX_CPE", "height")
        self.amax_cpe.elevation_min = config.getfloat("AMAX_CPE", "elevation_min")
        self.amax_cpe.elevation_max = config.getfloat("AMAX_CPE", "elevation_max")
        self.amax_cpe.azimuth = config.get("AMAX_CPE", "azimuth")
        self.amax_cpe.frequency = config.getfloat("AMAX_CPE", "frequency")
        self.amax_cpe.bandwidth = config.getfloat("AMAX_CPE", "bandwidth")
        self.amax_cpe.adjacent_ch_selectivity = config.getfloat("AMAX_CPE", "adjacent_ch_selectivity")
        self.amax_cpe.tx_power_density = config.getfloat("AMAX_CPE", "tx_power_density")
        self.amax_cpe.noise_temperature = config.getfloat("AMAX_CPE", "noise_temperature")
        self.amax_cpe.inr_scaling = config.getfloat("AMAX_CPE", "inr_scaling")
        self.amax_cpe.antenna_gain = config.getfloat("AMAX_CPE", "antenna_gain")
        self.amax_cpe.antenna_pattern = config.get("AMAX_CPE", "antenna_pattern")
        self.amax_cpe.antenna_envelope_gain = config.getfloat("AMAX_CPE", "antenna_envelope_gain")
        self.amax_cpe.diameter = config.getfloat("AMAX_CPE", "diameter")
        self.amax_cpe.channel_model = config.get("AMAX_CPE", "channel_model")
        self.amax_cpe.line_of_sight_prob = config.getfloat("AMAX_CPE", "line_of_sight_prob")
        self.amax_cpe.BOLTZMANN_CONSTANT = config.getfloat("AMAX_CPE", "BOLTZMANN_CONSTANT")
        self.amax_cpe.EARTH_RADIUS = config.getfloat("AMAX_CPE", "EARTH_RADIUS")
        self.amax_cpe.downtilt_deg = config.getfloat("AMAX_CPE", "downtilt_deg")
        self.amax_cpe.element_phi_deg_3db  = config.getfloat("AMAX_CPE", "amax_phi_deg_3db")
        self.amax_cpe.element_theta_deg_3db = config.getfloat("AMAX_CPE", "amax_theta_deg_3db")

        # P452 parameters
        self.amax_cpe.atmospheric_pressure = config.getfloat("AMAX_CPE", "atmospheric_pressure")
        self.amax_cpe.air_temperature = config.getfloat("AMAX_CPE", "air_temperature")
        self.amax_cpe.N0 = config.getfloat("AMAX_CPE", "N0")
        self.amax_cpe.delta_N = config.getfloat("AMAX_CPE", "delta_N")
        self.amax_cpe.percentage_p = config.get("AMAX_CPE", "percentage_p")
        self.amax_cpe.Dct = config.getfloat("AMAX_CPE", "Dct")
        self.amax_cpe.Dcr = config.getfloat("AMAX_CPE", "Dcr")
        self.amax_cpe.Hte = config.getfloat("AMAX_CPE", "Hte")
        self.amax_cpe.Hre = config.getfloat("AMAX_CPE", "Hre")
        self.amax_cpe.tx_lat = config.getfloat("AMAX_CPE", "tx_lat")
        self.amax_cpe.rx_lat = config.getfloat("AMAX_CPE", "rx_lat")
        self.amax_cpe.polarization = config.get("AMAX_CPE", "polarization")
        self.amax_cpe.clutter_loss = config.getboolean("AMAX_CPE", "clutter_loss")
        
        # Radar for indoor interference propagation parameters
        self.amax_cpe.es_position = config.get("AMAX_CPE", "es_position")
        self.amax_cpe.shadow_enabled = config.getboolean("AMAX_CPE", "shadow_enabled")
        self.amax_cpe.building_loss_enabled = config.getboolean("AMAX_CPE", "building_loss_enabled")
        self.amax_cpe.same_building_enabled = config.getboolean("AMAX_CPE", "same_building_enabled")
        self.amax_cpe.diffraction_enabled = config.getboolean("AMAX_CPE", "diffraction_enabled")
        self.amax_cpe.ap_building_entry_loss_type = config.get("AMAX_CPE", "ap_building_entry_loss_type")
        self.amax_cpe.ap_building_entry_loss_prob = config.getfloat("AMAX_CPE", "ap_building_entry_loss_prob")
        self.amax_cpe.ap_building_entry_loss_value = config.getfloat("AMAX_CPE", "ap_building_entry_loss_value")
