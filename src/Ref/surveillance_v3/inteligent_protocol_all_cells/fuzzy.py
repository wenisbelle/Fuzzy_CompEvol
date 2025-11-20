import numpy as np
from typing import Tuple, List
import random

import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyEvaluator:
    def __init__(self, map_width: int, map_height: int, camera_angle:float, individual:Tuple, distance_between_cells: int =10):
        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells
        
        ## Creating the fuzzy system without the ga optimization
        uncertainty = ctrl.Antecedent(np.arange(0, ))


    def get_cells_visited_in_trajectory(self, drone_altitude: float, initial_cell: Tuple[int, int], final_cell: Tuple[int, int]) -> list:
        
        cells_within_trajectory = []
        radius_coverage = drone_altitude * np.tan(self.camera_angle)
        
        if final_cell[0] == initial_cell[0]:  # Vertical line case
            for y in range(min(initial_cell[1], final_cell[1])+1, max(initial_cell[1], final_cell[1])+1):
                cells_within_trajectory.append((initial_cell[0], y))
            return cells_within_trajectory    
        elif final_cell[1] == initial_cell[1]:  # Horizontal line case
            for x in range(min(initial_cell[0], final_cell[0])+1, max(initial_cell[0], final_cell[0])+1):
                cells_within_trajectory.append((x, initial_cell[1]))
            return cells_within_trajectory
        else:
            line_slop = (final_cell[1] - initial_cell[1])/(final_cell[0] - initial_cell[0] + 1e-6)

            for y in range(min(initial_cell[1], final_cell[1])+1, max(initial_cell[1], final_cell[1])+1):
                for x in range(min(initial_cell[0], final_cell[0])+1, max(initial_cell[0], final_cell[0])+1):
                    ## distance from point to line formula ##
                    d = abs(line_slop*x - y + (initial_cell[1] -line_slop*initial_cell[0]))/np.sqrt(line_slop**2 + 1)
                    map_size_d = d * self.distance_between_cells
                    if map_size_d <= radius_coverage:
                        cells_within_trajectory.append((x, y))

            return cells_within_trajectory 