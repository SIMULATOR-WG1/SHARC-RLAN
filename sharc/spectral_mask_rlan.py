# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:06:56 2017

@author: Calil
"""

from sharc.support.enumerations import StationType
from sharc.spectral_mask import SpectralMask

import numpy as np
import math
import matplotlib.pyplot as plt

class SpectralMaskRlan(SpectralMask):
    """
    Implements spectral masks 
    
    Attributes:
        spurious_emissions (float): level of power emissions at spurious
            domain (dBm/MHz). -59 dBm/MHz for each channel. The 
            result could be compared as aggregated with -27 dBm/MHz,  
            as specified in document Anatel Resolução nº 680, de 27 de junho de 2017. 
            http://www.anatel.gov.br/legislacao/resolucoes/2017/936-resolucao-680
            and parameters acording:
            REQUISITOS TÉCNICOS PARA A AVALIAÇÃO DA CONFORMIDADE DE EQUIPAMENTOS DE RADIOCOMUNICAÇÃO DE RADIAÇÃO RESTRITA
            https://sei.anatel.gov.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=2549681&id_orgao_publicacao=0
        delta_f_lin (np.array): mask delta f breaking limits in MHz. Delta f 
            values for which the spectral mask changes value. Hard coded as
            [0,1000]. In this context, delta f is the frequency distance to
            the transmission's edge frequencies
        freq_lim (no.array): frequency values for which the spectral mask
            changes emission value
        sta_type (StationType): type of station to which consider the spectral
            mask. Possible values are StationType.RLAN_AP and StationType.RLAN_UE
        freq_mhz (float): center frequency of station in MHz
        band_mhs (float): transmitting bandwidth of station in MHz
        scenario (str): INDOOR or OUTDOOR scenario
        p_tx (float): station's transmit power in dBm/MHz
        mask_dbm (np.array): spectral mask emission values in dBm
    """
    def __init__(self,sta_type: StationType, freq_mhz: float, band_mhz: float, scenario = "OUTDOOR"):
        """
        Class constructor.
        
        Parameters:
            sta_type (StationType): type of station to which consider the spectral
                mask. Possible values are StationType.RLAN_AP and StationType.
                RLAN_UE
            freq_mhz (float): center frequency of station in MHz
            band_mhs (float): transmitting bandwidth of station in MHz
            scenario (str): INDOOR or OUTDOOR scenario
        """
        # Spurious domain limits [dBm/MHz]
        self.spurious_emissions = -29
        # Mask delta f breaking limits [MHz]
        self.delta_f_lim = np.array([0,0, 3, 41, 81])
        
        # Attributes
        self.sta_type = sta_type
        self.scenario = scenario
        self.band_mhz = band_mhz
        self.freq_mhz = freq_mhz
        
        self.freq_lim = np.concatenate(((freq_mhz - band_mhz/2)-self.delta_f_lim[::-1],freq_mhz,
                                        (freq_mhz + band_mhz/2)+self.delta_f_lim), axis=None)
       
        
    def set_mask(self, power = 0):
        """
        Sets the spectral mask (mask_dbm attribute) based on station type, 
        operating frequency and transmit power.
        
        Parameters:
            power (float): station transmit power in dBm. Default = 0
        """
        self.p_tx = power - 10*np.log10(self.band_mhz)
        
        # Set new transmit power value       
        if self.sta_type is StationType.RLAN_UE:
            # Table 8
            mask_dbm =  np.array([self.p_tx,self.p_tx -20,self.p_tx -28,self.p_tx -40, self.spurious_emissions])
            
        elif self.sta_type is StationType.RLAN_AP:             
            # Table 1
            mask_dbm =  np.array([self.p_tx,self.p_tx -20,self.p_tx -28,self.p_tx -40, self.spurious_emissions])
            
        self.mask_dbm = np.concatenate((mask_dbm[::-1],np.array([self.p_tx]),mask_dbm))
        
if __name__ == '__main__':
    # Initialize variables
    sta_type = StationType.RLAN_UE
    p_tx = 30.03  #in dBm for RLAN Wifi is 800 mw (10*log(800)= 29.03)
    freq = 5210
    band = 80
    
    # Create mask
    msk = SpectralMaskRlan(sta_type,freq,band)
    msk.set_mask(p_tx)
    
    # Frequencies
    freqs = np.linspace(-200,200,num=1000)+freq
    
    # Mask values
    mask_val = np.ones_like(freqs)*msk.mask_dbm[0]
    #for k in range(len(msk.freq_lim)-1,-1,-1):
#        mask_val[np.where(freqs < msk.freq_lim[k])] = msk.mask_dbm[k]
    mask_val = np.interp(freqs,msk.freq_lim,msk.mask_dbm)
        
    # Plot
    plt.plot(freqs,mask_val)
    plt.xlim([freqs[0],freqs[-1]])
    plt.xlabel("$\Delta$f [MHz]")
    plt.ylabel("Spectral Mask [dBm/MHz]")
    plt.grid()
    plt.show()
