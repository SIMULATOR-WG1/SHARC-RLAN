# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:17:03 2019

@author: Jussif
"""

import numpy as np
import sys

from sharc.support.named_tuples import AntennaPar

class AntennaElementLaar5a(object):
    """
    Implements a single element of an LAA antenna array following ITU-R Technical 
    characteristics and operational requirements of WAS/RLAN in the 5 GHz frequency 
    range considering item 3.5 and using parameters from ITU-R M2292

    Attributes
    ----------
        g_max (float): maximum gain of element
        theta_3db (float): vertical 3dB beamwidth of single element [degrees]
        phi_3db (float): horizontal 3dB beamwidth of single element [degrees]
        am (float): front-to-back ratio
        sla_v (float): element vertical sidelobe attenuation
    """

    def __init__(self,par: AntennaPar):
        """
        Constructs an AntennaElementRlan object.

        Parameters
        ---------
            param (ParametersAntennaRlan): antenna RLAN parameters
        """
        self.param = par

        self.g_max = par.element_max_g
        self.downtilt_rad = par.downtilt_deg / 180 * np.pi
        self.downtilt_deg = par.downtilt_deg 
        self.phi_deg_3db = par.element_phi_deg_3db
        self.sla= par.element_sla_v
        
        if par.element_theta_deg_3db > 0:
            self.theta_deg_3db = par.element_theta_deg_3db
        else:
            if self.phi_deg_3db > 120.:
                sys.stderr.write("ERROR\nvertical beamwidth must be givem if horizontal beamwidth > 120 degrees")
                sys.exit(1)
            # calculate based on F1336
            self.theta_deg_3db = (31000 * 10**(-.1 * self.g_max))/self.phi_deg_3db

        # antenna paremeters, according to ITU-R M2292
        self.k_a = .7
        self.k_p = .7
        self.k_h = .7
        self.lambda_k_h = 3 * (1-.5**(-self.k_h))
        self.k_v = .5
        self.incline_factor = np.log10(((180/self.theta_deg_3db)**1.5 * (4**-1.5+self.k_v))/
                                       (1 + 8 * self.k_p)) / np.log10(22.5 / self.theta_deg_3db)
        self.x_k = np.sqrt(1 - .36 * self.k_v)
        self.lambda_k_v = 12 - self.incline_factor * np.log10(4) - 10 * np.log10(4**-1.5 + self.k_v)

        self.g_hr_180 = -12. + 10 * np.log10(1 + 8 * self.k_a) - 15 * np.log10(180/self.theta_deg_3db)
        self.g_hr_0 = 0

    def horizontal_pattern(self, phi: np.array) -> {np.array, float}:
        """
        Calculates the horizontal radiation pattern for Omnidirectionnal patter.
        
        Parameters
        ----------
            phi (np.array): azimuth angle [degrees]
        Returns
        -------
            gain: horizontal radiation pattern gain value
        """

        gain = np.zeros(np.size(phi))


        return gain

    def vertical_pattern(self, theta: np.array) -> np.array:
        """
        Calculates the vertical radiation pattern.

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]

        Returns
        -------
            a_v (np.array): vertical radiation pattern gain value
        """
        
        gain = np.zeros(theta.shape)

        oper_1= 12*(((theta - self.downtilt_deg )/self.theta_deg_3db)**2)
        minimo= np.minimum(oper_1, self.sla)
        gain= self.g_max - minimo

        return gain

    def element_pattern(self, phi: np.array, theta: np.array) -> np.array:
        """
        Calculates the element radiation pattern gain.

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]
            phi (np.array): azimuth angle [degrees]

        Returns
        -------
            gain (np.array): element radiation pattern gain value
        """

        # recalculate angles considering mechanical tilt (eqs 3b/3c)
        theta_rad = -theta / 180 * np.pi
        phi_rad = phi / 180 * np.pi
        new_theta_rad = np.arcsin(np.sin(theta_rad) * np.cos(self.downtilt_rad) +
                                  np.cos(theta_rad) * np.cos(phi_rad) * np.sin(self.downtilt_rad))
        cos = (-np.sin(theta_rad) * np.sin(self.downtilt_rad) +
                np.cos(theta_rad) * np.cos(phi_rad) * np.cos(self.downtilt_rad))/np.cos(new_theta_rad)


        phi_rad = np.arccos(cos)
        theta = new_theta_rad / np.pi * 180
        phi = phi_rad / np.pi * 180

        gain_hor = self.horizontal_pattern(phi)
        compression_ratio = (gain_hor - self.g_hr_180)/(self.g_hr_0 - self.g_hr_180)
        gain = gain_hor + compression_ratio * self.vertical_pattern(theta)
        
#        gain = self.vertical_pattern(theta)
        
        return gain

if __name__ == '__main__':

    from sharc.parameters.parameters_antenna_rlan import ParametersAntennaRlan
    from matplotlib import pyplot as plt

    param = ParametersAntennaRlan()

    param.element_max_g = 5
    param.element_phi_deg_3db = 40
    param.element_theta_deg_3db = 40
    param.element_sla_v= 20

    #************************* x degrees tilt = -10 **************************#
    param.downtilt_deg = -10

    antenna = AntennaElementLaar5a( param )

    phi_vec = np.arange(-180,180, step = 40)
    theta_vec = np.arange(0,90, step = 5)

    #variables for:
    #horizontal pattern
    pattern_hor_0deg = np.zeros(phi_vec.shape)
    pattern_hor_10deg = np.zeros(phi_vec.shape)
    pattern_hor_30deg = np.zeros(phi_vec.shape)
    pattern_hor_60deg = np.zeros(phi_vec.shape)
    pattern_hor_90deg = np.zeros(phi_vec.shape)

    #vertical pattern
    pattern_ver_0deg = np.zeros(theta_vec.shape)
    pattern_ver_30deg = np.zeros(theta_vec.shape)
    pattern_ver_60deg = np.zeros(theta_vec.shape)
    pattern_ver_90deg = np.zeros(theta_vec.shape)
    
    #loop for horizontal pattern (azimuth)
    for phi, index in zip(phi_vec, range(len(phi_vec))):
        pattern_hor_0deg[index] = antenna.element_pattern(phi,0)
        pattern_hor_10deg[index] = antenna.element_pattern(phi, 10)
        pattern_hor_30deg[index] = antenna.element_pattern(phi, 30)
        pattern_hor_60deg[index] = antenna.element_pattern(phi, 60)
        pattern_hor_90deg[index] = antenna.element_pattern(phi, 90)

    plt.figure(1)
    plt.plot(phi_vec, pattern_hor_0deg, label = 'elevation = 0 degrees')
    plt.plot(phi_vec, pattern_hor_10deg, label = 'elevation = 10 degrees')
    plt.plot(phi_vec, pattern_hor_30deg, label = 'elevation = 30 degrees')
    plt.plot(phi_vec, pattern_hor_60deg, label = 'elevation = 60 degrees')
    plt.plot(phi_vec, pattern_hor_60deg, label = 'elevation = 90 degrees')

    plt.title('downtilt = -10 degrees')
    plt.xlabel ('azimuth (degrees)')
    plt.ylabel ('gain (dBi)')
    plt.grid()
    plt.legend()
    
    
    #loop for vertical pattern (elevation)
    for theta, index in zip(theta_vec, range(len(theta_vec))):
        pattern_ver_0deg[index] = antenna.element_pattern(0, theta)
        pattern_ver_30deg[index] = antenna.element_pattern(30, theta)
        pattern_ver_60deg[index] = antenna.element_pattern(60, theta)
        pattern_ver_90deg[index] = antenna.element_pattern(90, theta)

    plt.figure(2)
    plt.plot(theta_vec, pattern_ver_0deg, label='azimuth = 0 degrees')
    plt.plot(theta_vec, pattern_ver_30deg, label='azimuth = 30 degrees')
    plt.plot(theta_vec, pattern_ver_60deg, label='azimuth = 60 degrees')
    plt.plot(theta_vec, pattern_ver_90deg, label='azimuth = 90 degrees')


    plt.title('downtilt = -10 degrees')
    plt.xlabel('elevation (degrees)')
    plt.ylabel('gain (dBi)')
    
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 90, step=10))
    ax.legend()
    
    plt.grid()
    plt.legend()

