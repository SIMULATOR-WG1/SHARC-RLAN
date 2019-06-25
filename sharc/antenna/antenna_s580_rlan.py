# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:11:32 2019

@author: Jussif
"""

from sharc.antenna.antenna import Antenna
from sharc.parameters.parameters_fss_es import ParametersFssEs

import numpy as np

class AntennaS580_rlan(Antenna):
    """
    Implements the Rlan antenna pattern according to Recommendation ITU-R S.580-6
    """

    def __init__(self, param: ParametersFssEs):
        super().__init__()
        self.peak_gain = param.antenna_gain
        lmbda = 3e8 / ( param.frequency * 1e6 )

        self.phi_min = 1
        if 100 * lmbda / param.diameter > 1:
           self.phi_min = 100 * lmbda / param.diameter

    def calculate_gain(self, *args, **kwargs) -> np.array:
        phi = np.absolute(kwargs["off_axis_angle_vec"])

        gain = np.zeros(phi.shape)

        idx_0 = np.where(phi < self.phi_min)[0]
        gain[idx_0] = self.peak_gain

        #ITU-R Recommend. S580-6. Recommends 1) 
        idx_1 = np.where((self.phi_min <= phi) & (phi <= 20))[0]
        gain[idx_1] = 29 - 25 * np.log10(phi[idx_1])
        
        #ITU-R Recommend. S580-6. Note 5) 
        idx_2 = np.where((20 < phi) & (phi <= 26.3))[0]
        gain[idx_2] = -3.5
        
        #ITU-R Recommend. 465-6  Recommends 2)
        idx_3 = np.where((26.3 < phi) & (phi < 48))[0]
        gain[idx_3] = 32 - 25 * np.log10(phi[idx_3])
        
        #ITU-R Recommend. 465-6 Recommends 2)
        idx_4 = np.where((48 <= phi) & (phi <= 180))[0]
        gain[idx_4] = -10

        return gain


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #phi = np.linspace(0.1, 100, num = 100000)
    phi = np.linspace(2, 90, num = 100000) 

    # initialize antenna 
    
    #Test 1, AMT EMBRAER
    param27 = ParametersFssEs()
    param27.antenna_pattern = "ITU-R S.580-6"
    param27.frequency = 5100
    param27.antenna_gain = 40
    param27.diameter = 2.4
    antenna27 = AntennaS580_rlan(param27)
    gainEM = antenna27.calculate_gain(phi_vec=phi)

    
    #Test 2, AMT IPEV
    param = ParametersFssEs()
    param.antenna_pattern = "ITU-R S.580-6"
    param.frequency = 5100
    param.antenna_gain = 30
    param.diameter = 2.4
    antenna = AntennaS580_rlan(param)
    gainIP = antenna.calculate_gain(phi_vec=phi)
    
    
    #Plotting...
    fig = plt.figure(figsize=(8,7), facecolor='w', edgecolor='k')  # create a figure object

    plt.semilogx(phi, gainEM, "-g", label = "EMBRAER")
    plt.semilogx(phi, gainIP, "-r", label = "IPEV")
#    plt.semilogx(phi, gainEM - param27.antenna_gain, "-b", label = "$f = 5.100$ $GHz,$ $D = 5$ $m$")

    plt.title("ITU-R S.580 antenna radiation pattern for RLAN")
    plt.xlabel("Off-axis angle $\phi$ [deg]")
    plt.ylabel("Gain relative to $G_m$ [dB]")
    plt.legend(loc="lower left")
    plt.xlim((phi[0], phi[-1]))
#    plt.ylim((-80, 80))

    ay = plt.gca()
    ay.set_yticks(np.arange(-30, 60, step=10))
    ay.legend()
    plt.grid()

    #phi in degrees

    fig = plt.figure(figsize=(8,7), facecolor='w', edgecolor='k')  # create a figure object

    plt.plot(phi, gainEM, "-g", label = "EMBRAER")
    plt.plot(phi, gainIP, "-r", label = "IPEV")

    plt.title("ITU-R S.580 antenna radiation pattern for RLAN")
    plt.xlabel("Off-axis angle $\phi$ [deg]")
    plt.ylabel("Gain relative to $G_m$ [dB]")
    plt.legend(loc="upper right")

#    ay = plt.gca()
#    ay.set_yticks(np.arange(-30, 60, step=10))
#    ax = plt.gca()
#    ax.set_xticks(np.arange(0, 90, step=10))
#    ax.legend()

    plt.grid()
    plt.show()
