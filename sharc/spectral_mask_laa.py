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

class SpectralMaskLaa(SpectralMask):
    """
    Implements spectral masks . The masks are in the
    document's .
    
    Attributes:
        spurious_emissions (float): level of power emissions at spurious
            domain (dBm/MHz). Hardcoded as -27 dBm/MHz,  as specified in
            document ITU 265-E
        delta_f_lin (np.array): mask delta f breaking limits in MHz. Delta f 
            values for which the spectral mask changes value. Hard coded as
            [0, 20, 400]. In this context, delta f is the frequency distance to
            the transmission's edge frequencies
        freq_lim (no.array): frequency values for which the spectral mask
            changes emission value
        sta_type (StationType): type of station to which consider the spectral
            mask. Possible values are StationType.LAA_BS and StationType.LAA_UE
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
        self.spurious_emissions = -27
        # Mask delta f breaking limits [MHz]
        self.delta_f_lim = np.array([0,0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8,9.5,10,11,12,13,14,15,16,17,18, 19.5, 40])
        
        # Attributes
        self.sta_type = sta_type
        self.scenario = scenario
        self.band_mhz = band_mhz
        self.freq_mhz = freq_mhz
        
        self.freq_lim = np.concatenate(((freq_mhz - band_mhz/2)-self.delta_f_lim[::-1],
                                        (freq_mhz + band_mhz/2)+self.delta_f_lim))
       
        
    def set_mask(self, power = 0):
        """
        Sets the spectral mask (mask_dbm attribute) based on station type, 
        operating frequency and transmit power.
        
        Parameters:
            power (float): station transmit power. Default = 0
        """
        self.p_tx = power - 10*np.log10(self.band_mhz)
        
        # Set new transmit power value       
        if self.sta_type is StationType.RLAN_UE:
            # Table 6.6.2.2.6-1  form TS.36.101
                   
           mask_dbm = np.ones_like(self.delta_f_lim)
           j=0
           for i in self.delta_f_lim:            
               if i < 1 :
                   mask_dbm[j] = 10-20*(i+0.5)
               elif  i >= 1 and i < 9.5 :
                   mask_dbm[j] = -10-8/9*(i+0.5-1)
               elif i >= 9.5 and i < 19.5 :
                   mask_dbm[j] = -18-1.2*(i+0.5-10)
               else : 
                   mask_dbm[j] = -30 
               j = j+1
                          
        elif self.sta_type is StationType.RLAN_AP:             
            # Table 6.6.3.2D-1  form TS.36.104
            # Formulas are valido for dBm/100KHz
           mask_dbm = np.ones_like(self.delta_f_lim)
           j=0
           for i in self.delta_f_lim:            
               if i < 1 :
                   mask_dbm[j] = self.p_tx-32.6-10*(i+0.05-0.05)+10
                   # Added +10 dB to convert to dbm/MHz
               elif  i >= 1 and i < 9.5 :
                   mask_dbm[j] = self.p_tx-42.6-8/9*(i+0.05-1.05)+10
               elif i >= 9.5 and i < 19.5 :
                   mask_dbm[j] = self.p_tx-50.6-12/10*(i+0.05-10.05)+10
               else : 
                   mask_dbm[j] = -40+10
               j = j+1
 
        self.mask_dbm = np.concatenate((mask_dbm[::-1],np.array([self.p_tx]),
                                        mask_dbm))
        
if __name__ == '__main__':
    # Initialize variables
    sta_type = StationType.LAA_UE
    p_tx = 17
    freq = 5250
    band = 20
    
    # Create mask
    msk = SpectralMaskLaa(sta_type,freq,band)
    msk.set_mask(p_tx)
    
    # Frequencies
    freqs = np.linspace(-50,50,num=1000)+freq
    
    # Mask values
    mask_val = np.ones_like(freqs)*msk.mask_dbm[0]
    for k in range(len(msk.freq_lim)-1,-1,-1):
        mask_val[np.where(freqs < msk.freq_lim[k])] = msk.mask_dbm[k]
        
    # Plot
    plt.plot(freqs,mask_val)
    plt.xlim([freqs[0],freqs[-1]])
    plt.xlabel("$\Delta$f [MHz]")
    plt.ylabel("Spectral Mask [dBm]")
    plt.grid()
    plt.show()
