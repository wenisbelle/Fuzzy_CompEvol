import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

class FitnessEvaluator:
    def __init__(self, map_width: int,
                 map_height: int,
                 distance_between_cells: int,
                 number_of_cells_x_y: int = 10):

        self.map_width = map_width
        self.map_height = map_height
        self.distance_between_cells = distance_between_cells
        self.NUMBER_OF_CELLS_X_Y = number_of_cells_x_y

    def check_maximum_cells(self, reduced_map: np.array):
        max_knowledge = reduced_map[:, :].max()
        rows, cols = np.where(reduced_map[:, :] == max_knowledge)
        return [list(zip(rows, cols)), max_knowledge]
    

    def choose_one_cell(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> list:
        """
        Batches inputs and uses the Interpolator instead of skfuzzy.compute
        """
        map_data = map_data.copy()    
             
        maximum_cells_coords, max_value = self.check_maximum_cells(map_data)

        if maximum_cells_coords:
            target_coords = random.choice(maximum_cells_coords)

        return [target_coords, max_value]

    def choose_two_cells(self, map_data: np.array, first_drone_pos, second_drone_pos, map_center_offset) -> list:
        map_data = map_data.copy()        

        ##### First Drone
        first_drone_maximum_cells_coords, value = self.check_maximum_cells(map_data)
        first_target_coords = random.choice(first_drone_maximum_cells_coords)

        ##### Second Drone   
        second_drone_maximum_cells_coords, second_drone_value = self.check_maximum_cells(map_data)
        second_target_coords = random.choice(second_drone_maximum_cells_coords)

        return [[first_target_coords, second_target_coords], second_drone_value]  

