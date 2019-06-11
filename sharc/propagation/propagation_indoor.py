# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:45:50 2017

@author: edgar
"""

from sharc.parameters.parameters_indoor import ParametersIndoor
from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_inh_office import PropagationInhOffice
from sharc.propagation.propagation_building_entry_loss import PropagationBuildingEntryLoss

import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

class PropagationIndoor(Propagation):
    """
    This is a wrapper class which can be used for indoor simulations. It
    calculates the basic path loss between AP's and UE's of the same building,
    assuming 3 AP's per building. It also includes an additional building
    entry loss for the outdoor UE's that are served by indoor AP's.
    """

    # For AP and UE that are not in the same building, this value is assigned
    # so this kind of inter-building interference will not be effectivelly
    # taken into account during SINR calculations. This is assumption
    # simplifies the implementation and it is reasonable: intra-building
    # interference is much higher than inter-building interference
    HIGH_PATH_LOSS = 400

    def __init__(self, random_number_gen: np.random.RandomState, param: ParametersIndoor,ue_per_cell):
        super().__init__(random_number_gen)

        if param.basic_path_loss == "FSPL":
            self.bpl = PropagationFreeSpace(random_number_gen)
        elif param.basic_path_loss == "INH_OFFICE":
            self.bpl = PropagationInhOffice(random_number_gen)
        else:
            sys.stderr.write("ERROR\nInvalid indoor basic path loss model: " + param.basic_path_loss)
            sys.exit(1)

        self.bel = PropagationBuildingEntryLoss(random_number_gen)
        self.building_class = param.building_class
        self.ap_per_building = param.num_cells
        self.ue_per_building = ue_per_cell*param.num_cells

    def get_loss(self, *args, **kwargs) -> np.array:
        """
        Calculates path loss for LOS and NLOS cases with respective shadowing
        (if shadowing has to be added)

        Parameters
        ----------
            distance_3D (np.array) : 3D distances between stations
            distance_2D (np.array) : 2D distances between stations
            elevation (np.array) : elevation angles from UE's to AP's
            frequency (np.array) : center frequencie [MHz]
            indoor (np.array) : indicates whether UE is indoor
            shadowing (bool) : if shadowing should be added or not

        Returns
        -------
            array with path loss values with dimensions of distance_2D

        """
        distance_3D = kwargs["distance_3D"]
        distance_2D = kwargs["distance_2D"]
        elevation = kwargs["elevation"]
        frequency = kwargs["frequency"]
        indoor = kwargs["indoor_stations"]
        shadowing = kwargs["shadowing"]

        loss = PropagationIndoor.HIGH_PATH_LOSS*np.ones(frequency.shape)
        iter = int(frequency.shape[0]/self.ap_per_building)
        for i in range(iter):
            bi = int(self.ap_per_building*i)
            bf = int(self.ap_per_building*(i+1))
            ui = int(self.ue_per_building*i)
            uf = int(self.ue_per_building*(i+1))

            # calculate basic path loss
            loss[bi:bf,ui:uf] = self.bpl.get_loss(distance_3D = distance_3D[bi:bf,ui:uf],
                                                  distance_2D = distance_2D[bi:bf,ui:uf],
                                                  frequency = frequency[bi:bf,ui:uf],
                                                  indoor = indoor[0,ui:uf],
                                                  shadowing = shadowing)

            # calculates the additional building entry loss for outdoor UE's
            # that are served by indoor AP's
            bel = (~ indoor[0,ui:uf]) * self.bel.get_loss(frequency[bi:bf,ui:uf], elevation[bi:bf,ui:uf], "RANDOM", self.building_class)

            loss[bi:bf,ui:uf] = loss[bi:bf,ui:uf] + bel

        return loss


if __name__ == '__main__':
    params = ParametersIndoor()
    params.basic_path_loss = "INH_OFFICE"
    params.n_rows = 3
    params.n_colums = 1
#    params.street_width = 30
    params.ue_indoor_percent = .95
    params.num_cells = 3
    params.building_class = "TRADITIONAL"

    ap_per_building = 3
    ue_per_ap = 3

    num_ap = ap_per_building*params.n_rows*params.n_colums
    num_ue = num_ap*ue_per_ap
    distance_2D = 150*np.random.random((num_ap, num_ue))
    frequency = 27000*np.ones(distance_2D.shape)
    indoor = np.random.rand(num_ap) < params.ue_indoor_percent
    h_ap = 3*np.ones(num_ap)
    h_ue = 1.5*np.ones(num_ue)
    distance_3D = np.sqrt(distance_2D**2 + (h_ap[:,np.newaxis] - h_ue)**2)
    height_diff = np.tile(h_ap, (num_ap, 3)) - np.tile(h_ue, (num_ap, 1))
    elevation = np.degrees(np.arctan(height_diff/distance_2D))

    propagation_indoor = PropagationIndoor(np.random.RandomState(),params,ue_per_ap)
    loss_indoor = propagation_indoor.get_loss(distance_3D = distance_3D,
                                              distance_2D = distance_2D,
                                              elevation = elevation,
                                              frequency = frequency,
                                              indoor_stations = indoor,
                                              shadowing = False)



