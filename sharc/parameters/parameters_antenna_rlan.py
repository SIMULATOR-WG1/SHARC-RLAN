# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:29:36 2017

@author: Calil
"""

from sharc.support.named_tuples import AntennaPar
from numpy import load

class ParametersAntennaRlan(object):
    """
    Defines parameters for antenna array.
    """

    def __init__(self):
        pass


    ###########################################################################
    # Named tuples which contain antenna types

    def get_antenna_parameters(self,sta_type: str, txrx: str)-> AntennaPar:
        if sta_type == "AP":
            
            if self.normalization:
                # Load data, save it in dict and close it
                data = load(self.ap_normalization_file)
                data_dict = {key:data[key] for key in data}
                self.ap_normalization_data = data_dict
                data.close()
                # Same for UE
                data = load(self.ue_normalization_file)
                data_dict = {key:data[key] for key in data}
                self.ue_normalization_data = data_dict
                data.close()
            else:
                self.ap_normalization_data = None
                self.ue_normalization_data = None
            
            if txrx == "TX":
                tpl = AntennaPar(self.normalization,
                                 self.ap_normalization_data,
                                 self.ap_element_pattern,
                                 self.ap_tx_element_max_g,
                                 self.ap_tx_element_phi_deg_3db,
                                 self.ap_tx_element_theta_deg_3db,
                                 self.ap_tx_element_am,
                                 self.ap_tx_element_sla_v,
                                 self.ap_tx_n_rows,
                                 self.ap_tx_n_columns,
                                 self.ap_tx_element_horiz_spacing,
                                 self.ap_tx_element_vert_spacing,
                                 self.ap_downtilt_deg)
            elif txrx == "RX":
                tpl = AntennaPar(self.normalization,
                                 self.ap_normalization_data,
                                 self.ap_element_pattern,
                                 self.ap_rx_element_max_g,
                                 self.ap_rx_element_phi_deg_3db,
                                 self.ap_rx_element_theta_deg_3db,
                                 self.ap_rx_element_am,
                                 self.ap_rx_element_sla_v,
                                 self.ap_rx_n_rows,
                                 self.ap_rx_n_columns,
                                 self.ap_rx_element_horiz_spacing,
                                 self.ap_rx_element_vert_spacing,
                                 self.ap_downtilt_deg)
        elif sta_type == "UE":
            if txrx == "TX":
                tpl = AntennaPar(self.normalization,
                                 self.ue_normalization_data,
                                 self.ue_element_pattern,
                                 self.ue_tx_element_max_g,
                                 self.ue_tx_element_phi_deg_3db,
                                 self.ue_tx_element_theta_deg_3db,
                                 self.ue_tx_element_am,
                                 self.ue_tx_element_sla_v,
                                 self.ue_tx_n_rows,
                                 self.ue_tx_n_columns,
                                 self.ue_tx_element_horiz_spacing,
                                 self.ue_tx_element_vert_spacing,
                                 0)
            elif txrx == "RX":
                tpl = AntennaPar(self.normalization,
                                 self.ue_normalization_data,
                                 self.ue_element_pattern,
                                 self.ue_rx_element_max_g,
                                 self.ue_rx_element_phi_deg_3db,
                                 self.ue_rx_element_theta_deg_3db,
                                 self.ue_rx_element_am,
                                 self.ue_rx_element_sla_v,
                                 self.ue_rx_n_rows,
                                 self.ue_rx_n_columns,
                                 self.ue_rx_element_horiz_spacing,
                                 self.ue_rx_element_vert_spacing,
                                 0)

        return tpl
