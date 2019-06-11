# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:41:25 2017

@author: edgar
"""
import sys

from sharc.topology.topology import Topology
from sharc.topology.topology_macrocell import TopologyMacrocell
from sharc.topology.topology_hotspot import TopologyHotspot
from sharc.topology.topology_indoor import TopologyIndoor
from sharc.topology.topology_single_base_station import TopologySingleBaseStation
from sharc.parameters.parameters import Parameters

class TopologyFactory(object):
    
    @staticmethod
    def createTopology(parameters: Parameters) -> Topology:
        if parameters.rlan.topology == "SINGLE_BS":
            return TopologySingleBaseStation(parameters.rlan.intersite_distance*2/3, parameters.rlan.num_clusters)
        elif parameters.rlan.topology == "MACROCELL":
            return TopologyMacrocell(parameters.rlan.intersite_distance, parameters.rlan.num_clusters)
        elif parameters.rlan.topology == "HOTSPOT":
            return TopologyHotspot(parameters.hotspot, parameters.rlan.intersite_distance, parameters.rlan.num_clusters)
        elif parameters.rlan.topology == "INDOOR":
            return TopologyIndoor(parameters.indoor)
        else:
            sys.stderr.write("ERROR\nInvalid topology: " + parameters.rlan.topology)
            sys.exit(1)            
