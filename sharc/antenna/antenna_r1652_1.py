# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:11:32 2019

@author: Jussif
"""

from sharc.antenna.antenna import Antenna
from sharc.parameters.parameters_fss_es import ParametersFssEs

import numpy as np



class AntennaRadar(Antenna):
    """
    Implements the radar antenna pattern 
    according to [1] Recommendation ITU-R M.1652-1, Appendix 1 to annex 6
    """

    def __init__(self, param: ParametersFssEs):
        super().__init__()        
        self.r_gain = param.antenna_gain          #radar gain in dBi
        self.theta_min = 0

    def calculate_angle(self):
        #r_gain = np.absolute(kwargs["off_axis_angle_vec"]) #to enable during complete simulation scenario.
        rn_gain= self.r_gain
        
        #thetas...
        r_tetha_m = 0
        r_tetha_r = 0
        r_tetha_b = 0
      
        #Considering Table 7 from [1]        
        if (self.r_gain > 10):
        
            r_tetha_m = (50*(0.25*rn_gain + 7)**0.5)/(10**(rn_gain/20.0)) 
            
        #Table 7 from [1] VERY HIGH GAIN 
        if (self.r_gain > 48):
            r_tetha_r = 27.466*(10**((-0.3*rn_gain)/10.0))
            r_tetha_b = 48.0 

        #Table 7 from [1] HIGH GAIN
        if ((22 < self.r_gain) and (self.r_gain <= 48)):
            r_tetha_r= 250.0/(10**(rn_gain/20.0))
            r_tetha_b= 48.0
        
#       #Table 7 from [1] MEDIUM GAIN  
        if ((10 < self.r_gain) and (self.r_gain < 22)):
            r_tetha_r= 250.0/(10**(rn_gain/20.0))
            r_tetha_b= 131.8257*(10**((-rn_gain)/50.0))

        results_ag= [r_tetha_m, r_tetha_r, r_tetha_b]
        
        return results_ag

#    def calculate_gain(self, tetha):
    def calculate_gain(self, *args, **kwargs) -> np.array:
        tetha = np.absolute(kwargs["off_axis_angle_vec"])
        angle = AntennaRadar.calculate_angle(self)
        gain = np.zeros(tetha.shape)
        
        rn_gain= self.r_gain

        #VERY HIGH GAIN ANTENNAS Table 8 
        if (self.r_gain > 48):
            idx_0 = np.where((tetha >= self.theta_min) & (tetha < angle[0] ))[0]
            idx_1 = np.where((tetha >= angle[0]) & (tetha < angle[1]))[0]
            idx_2 = np.where((tetha >= angle[1]) & (tetha < angle[2]))[0]
            idx_3 = np.where((tetha >= angle[2]) & (tetha <= 180 ))[0]

            gain[idx_0] = rn_gain - 4*((10**(-4))*(10**(rn_gain/10.0))*(tetha[idx_0]**2))
            gain[idx_1] = 0.75*rn_gain - 7
            gain[idx_2] = 29 - 25 * np.log10(tetha[idx_2]) 
            gain[idx_3] = -13
        
        #HIGH GAIN ANTENNAS Table 9 
        if (self.r_gain > 22 and self.r_gain < 48):
            idx_0 = np.where((tetha >= self.theta_min) & (tetha < angle[0] ))[0]
            idx_1 = np.where((tetha >= angle[0]) & (tetha < angle[1] ))[0]
            idx_2 = np.where((tetha >= angle[1]) & (tetha < angle[2] ))[0]
            idx_3 = np.where((tetha >= angle[2]) & (tetha <= 180 ))[0]

            gain[idx_0] = rn_gain - 4*((10**(-4))*(10**(rn_gain/10.0))*(tetha[idx_0]**2))
            gain[idx_1] = 0.75*rn_gain - 7
            gain[idx_2] = 53 -(rn_gain/2.0) - 25*np.log10(tetha[idx_2])
            gain[idx_3] = 11 -(rn_gain/2.0)

        #MEDIUM GAIN ANTENNAS Table 10
        if (self.r_gain > 10 and self.r_gain < 22):
            idx_0 = np.where((tetha >= self.theta_min) & (tetha < angle[0] ))[0]
            idx_1 = np.where((tetha >= angle[0]) & (tetha < angle[1] ))[0]
            idx_2 = np.where((tetha >= angle[1]) & (tetha < angle[2] ))[0]
            idx_3 = np.where((tetha >= angle[2]) & (tetha <= 180 ))[0]
    
            gain[idx_0] = rn_gain - 4*((10**(-4))*(10**(rn_gain/10.0))*(tetha[idx_0]**2))
            gain[idx_1] = 0.75*rn_gain - 7
            gain[idx_2] = 53 -(rn_gain/2.0) - 25*np.log10(tetha[idx_2])
            gain[idx_3] = 0

        return gain

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #theta angle.. 
    tetha = np.linspace(0, 180, num = 100000) 
    
    #radar antenna gain dbi
    #r_gain = np.linspace(10, 60, num = 100000)
    
    
    #Test 1, Radar
    param27 = ParametersFssEs()
    param27.antenna_pattern = "ITU-R M.1652-1"
    param27.frequency = 5100
    param27.antenna_gain = 35
    param27.diameter = 5
    antenna27 = AntennaRadar(param27)
#    resultado_angle27 = antenna27.calculate_angle()
    gain27 = antenna27.calculate_gain(tetha)


    #Plotting...
    
    plt.plot(tetha, gain27, "-b", label = param27.antenna_gain )
#    plt.semilogx(r_tetha, gain27 - param27.antenna_gain, "-b", label = "$f = 5.100$ $GHz,$ $D = 5$ $m$")
    #plt.semilogx(phi, gain - param.antenna_gain, "-r", label = "$f = 5.100$ $GHz,$ $D = 0.45$ $m$")

    plt.title("ITU-R M.1652-1 radar antenna radiation pattern")
    plt.xlabel("Off-axis angle $\phi$ [deg]")
    plt.ylabel("Gain relative to $G_m$ [dBi]")
    plt.legend(loc="lower left")


    ax = plt.gca()
    #ax.set_yticks([-30, -20, -10, 0])
#    ax.set_xticks(np.linspace(1, 9, 9).tolist() + np.linspace(10, 100, 10).tolist())
    ax.set_xticks(np.arange(0, 190, step=20))
    ax.legend()
    
    plt.grid()
    plt.show()
    