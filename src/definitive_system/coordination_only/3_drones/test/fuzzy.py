import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import RegularGridInterpolator

class FuzzyEvaluator:
    def __init__(self, map_width: int,
                 map_height: int,
                 camera_angle:float,
                 fuzzy_tables: List[RegularGridInterpolator],
                 distance_between_cells: int =10):

        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells

        self.NUMBER_OF_CELLS_X_Y = 10

        # Verify all fuzzy tables are not None
        if any(t is None for t in fuzzy_tables):
            raise ValueError("All fuzzy tables must be properly initialized")

        self.interp_one_cell = fuzzy_tables[0]
        self.interp_two_cells = fuzzy_tables[1] 
     

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

    def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> list:
        """
        Batches inputs and uses the Interpolator instead of skfuzzy.compute
        """
        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        
        current_i = int((drone_x + map_center_offset)/self.distance_between_cells)
        current_j = int((drone_y + map_center_offset)/self.distance_between_cells)

        # 1. Collect inputs in lists (Geometric part is still a loop, but lightweight)
        input_data = [] # Will hold tuples of (uncertainty, distance, ind_uncertainty)
        coords_tracker = []

        rows, cols = map_data.shape

        # Max/min bounds ensure we don't look outside the map array
        start_i = max(0, current_i - self.NUMBER_OF_CELLS_X_Y//2)
        end_i = min(rows, current_i + self.NUMBER_OF_CELLS_X_Y//2)
        
        start_j = max(0, current_j - self.NUMBER_OF_CELLS_X_Y//2)
        end_j = min(cols, current_j + self.NUMBER_OF_CELLS_X_Y//2)
        
        
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                # Distance
                x_cell = self.distance_between_cells*i - map_center_offset
                y_cell = self.distance_between_cells*j - map_center_offset
                dist = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)

                trajectory_cells = self.get_cells_visited_in_trajectory(
                    drone_altitude=drone_position[2],
                    initial_cell=(current_i, current_j),
                    final_cell=(i, j)
                )
                average_trajectory_cells = sum([map_data[cell[0], cell[1]] for cell in trajectory_cells])/len(trajectory_cells) if trajectory_cells else 0.0
                
                trajectory_value = np.clip(average_trajectory_cells, 0, 30.0)
                input_data.append([trajectory_value, dist])
                coords_tracker.append((i, j))

        input_array = np.array(input_data)
        
        priorities = self.interp_one_cell(input_array)

        priority_scores = []
        for idx, score in enumerate(priorities):
            priority_scores.append((score, coords_tracker[idx]))
            
        return priority_scores

    def both_cells_priority(self, map_data: np.array, first_drone_pos, second_drone_pos, map_center_offset) -> list:
        """
        Fully vectorized fuzzy inference for two drones.
        """
        # Get individual priorities (using the fast method above)
        list1 = self.cells_priority(map_data, first_drone_pos, map_center_offset)
        list2 = self.cells_priority(map_data, second_drone_pos, map_center_offset)

        if not list1 or not list2:
            return []

        # Convert to arrays for vectorization
        p1_vals = np.array([x[0] for x in list1])
        p1_coords = np.array([x[1] for x in list1]) # Shape (N, 2)

        p2_vals = np.array([x[0] for x in list2])
        p2_coords = np.array([x[1] for x in list2]) # Shape (M, 2)

        # 1. Vectorized Sum of Priorities
        # Shape (N, 1) + (1, M) -> (N, M)
        sum_p_matrix = p1_vals[:, np.newaxis] + p2_vals[np.newaxis, :]

        #Vectorized Distance Calculation
        # Convert grid indices to physical coordinates
        phys_p1 = (p1_coords * self.distance_between_cells) - map_center_offset
        phys_p2 = (p2_coords * self.distance_between_cells) - map_center_offset

        # Broadcasting distance: (N, 1, 2) - (1, M, 2)
        diff = phys_p1[:, np.newaxis, :] - phys_p2[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2)) # Shape (N, M)

        #Prepare inputs for Interpolator     
        flat_sum = sum_p_matrix.ravel()
        flat_dist = dist_matrix.ravel()
        
        input_stack = np.column_stack((flat_sum, flat_dist))

        combined_priorities = self.interp_two_cells(input_stack)

        #Reconstruct the list structure
        # We need indices to know which cells generated which priority
        n_indices, m_indices = np.indices(sum_p_matrix.shape)
        flat_n = n_indices.ravel()
        flat_m = m_indices.ravel()

        combined_priority_scores = []
        for k in range(len(combined_priorities)):
             cell1 = list1[flat_n[k]][1]
             cell2 = list2[flat_m[k]][1]
             score = combined_priorities[k]
             combined_priority_scores.append((score, cell1, cell2))

        return combined_priority_scores
    
    def choose_one_cell(self, fitness_scores: list) -> Tuple[float, float]:
        if not fitness_scores:
            return None
        
        best_cell = max(fitness_scores, key=lambda x: x[0])
        # Return the coordinates
        return [best_cell[1], best_cell[0]]
    
    def choose_two_cells(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        fitness_scores = max(fitness_scores, key=lambda x: x[0])
        best_1 = (fitness_scores[1])
        best_2 = (fitness_scores[2])

        return [[best_1, best_2], fitness_scores[0]]

    
##### For sanity checks purposes only #####
"""
def main():
    sample_fuzzy_parameters_fixed = np.array([
    3.0, 7.0, 12.0,     # uncertainty (3 values)
    40.0, 80.0, 120.0,  # distance (3 values)
    0.5, 1.0, 1.5,      # individual_cell_uncertainty (3 values)    
    # one_cell_priority_interval (4 values: [A, B, C, D] from original code):
    0.25, 0.5, 0.75     # This allows your current extraction fuzzy_parameters[9:13] to work.
])
    evaluator = FuzzyEvaluator(map_width=10, map_height=10, camera_angle=np.radians(30), fuzzy_parameters=sample_fuzzy_parameters_fixed)
    
if __name__ == "__main__":
    main()
"""