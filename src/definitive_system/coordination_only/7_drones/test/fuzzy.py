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
                 number_of_cells_x_y: int = 10, 
                 distance_between_cells: int =10):

        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells

        self.NUMBER_OF_CELLS_X_Y = number_of_cells_x_y

        # Verify all fuzzy tables are not None
        if any(t is None for t in fuzzy_tables):
            raise ValueError("All fuzzy tables must be properly initialized")

        self.interp_one_cell = fuzzy_tables[0]
        self.interp_two_cells = fuzzy_tables[1] 
     

    def get_cells_visited_in_trajectory(self, drone_altitude: float, initial_cell: Tuple[int, int], final_cell: Tuple[int, int]) -> list:
        
        x0, y0 = initial_cell
        x1, y1 = final_cell

        radius_coverage = drone_altitude * np.tan(self.camera_angle)

        x_min, x_max = min(x0, x1) , max(x0, x1) 
        y_min, y_max = min(y0, y1) , max(y0, y1) 

        X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))

        ### Equation distance between point and line
        A = y1 - y0
        B = x0 - x1
        C = x1 * y0 - y1 * x0
        denominator = np.sqrt(A**2 + B**2)

        if denominator == 0:
            # Drone is evaluating its exact current cell
            return [(x0, y0)]
        
        distances = np.abs(A * X + B * Y + C) / denominator
        map_size_d = distances * self.distance_between_cells

        # Filter cells within the camera radius
        mask = map_size_d <= radius_coverage
        
        # Extract the valid coordinates
        valid_x = X[mask]
        valid_y = Y[mask]
        
        # Filter out coordinates that fall outside the actual map boundaries
        bounds_mask = (valid_x >= 0) & (valid_x < self.map_width) & (valid_y >= 0) & (valid_y < self.map_height)
        valid_x = valid_x[bounds_mask]
        valid_y = valid_y[bounds_mask]
        
        # Create the final list of absolute coordinates
        cells_within_trajectory = list(zip(valid_x, valid_y))

        return cells_within_trajectory       

    def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> list:
        """
        Batches inputs and uses the Interpolator instead of skfuzzy.compute
        """
        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        
        current_i = int((drone_x + map_center_offset)/self.distance_between_cells)
        current_j = int((drone_y + map_center_offset)/self.distance_between_cells)

        input_data = [] # Will hold tuples of (uncertainty, distance, ind_uncertainty)
        coords_tracker = []

        rows, cols = map_data.shape

        min_x_cell = max(0, current_i - self.NUMBER_OF_CELLS_X_Y//2)
        max_x_cell = min(rows, current_i + self.NUMBER_OF_CELLS_X_Y//2)
        min_y_cell = max(0, current_j - self.NUMBER_OF_CELLS_X_Y//2)
        max_y_cell = min(cols, current_j + self.NUMBER_OF_CELLS_X_Y//2)

        print(f"Evaluating cells in range x: [{min_x_cell}, {max_x_cell}), y: [{min_y_cell}, {max_y_cell}) for drone at position ({drone_x}, {drone_y}) with current cell ({current_i}, {current_j})")
        
        for i in range(min_x_cell, max_x_cell):
            for j in range(min_y_cell, max_y_cell):
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
            priority_scores.append((score, coords_tracker[idx], input_data[idx][0], input_data[idx][1])) # (priority_score, (i,j), trajectory_value)
            
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
        print(f"Best cell: {best_cell[1]} with priority score: {best_cell[0]} and trajectory value: {best_cell[2]} and distance: {best_cell[3]}")
        # Return the coordinates
        return [best_cell[1], best_cell[0]]
    
    def choose_two_cells(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        fitness_scores = max(fitness_scores, key=lambda x: x[0])
        best_1 = (fitness_scores[1])
        best_2 = (fitness_scores[2])

        return [[best_1, best_2], fitness_scores[0]]

