# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:53:41 2017

@author: edgar
"""

from enum import Enum

class Action( Enum ):
    """
    The action that is sent to controller in order to control the simulation
    """
    START_SIMULATION = 1
    START_SIMULATION_SINGLE_THREAD = 2
    STOP_SIMULATION  = 3
    
class State( Enum ):
    """
    This is the graphical user interface state
    """
    INITIAL  = 1
    RUNNING  = 2
    FINISHED = 3
    STOPPED  = 4
    STOPPING = 5
    
class StationType(Enum):
    """
    Station types supported by simulator.
    """
    NONE   = 0  # Dummy enum, for initialization purposes only
    RLAN_AP = 1  # RLAN Base Station
    RLAN_UE = 2  # RLAN User Equipment
    FSS_SS = 3  # FSS Space Station
    FSS_ES = 4  # FSS Earth Station
    FS     = 5  # Fixed Service
    HAPS   = 6  # HAPS (airbone) station
    RNS    = 7  # Radionavigation service
    RAS    = 8  # Radio Astronomy Service
    AMT_GS = 9  # AMT Ground Station
    RDR_GS = 10 # Radar Ground Station
    AMAX_BS = 11 # Radar Ground Station
    AMAX_CPE = 12 # Radar Ground Station
    LAA_BS = 13 # LAA Base Station
    LAA_UE = 14 # LAA User Equipment
    
