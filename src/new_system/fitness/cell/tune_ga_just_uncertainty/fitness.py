import numpy as np
from collections import deque
from typing import Tuple, List
import random
import heapq


class FitnessEvaluator:
    def __init__(self, map_width: int, map_height: int, camera_angle:float, distance_between_cells: int =10):
        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells

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
        

    def cells_fitness(self, map_data: np.array,
                      drone_position: Tuple[float, float, float],
                      distance_norm: float,
                      average_trajectory_accomulate_fitness_norm: float,
                      map_center_offset: float,
                      distance_between_cells: int) -> list:

        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        #### Getting the current index of the drone in the map ####
        current_i = int((drone_x + map_center_offset)/distance_between_cells)
        current_j = int((drone_y + map_center_offset)/distance_between_cells)
        #print(f"Drone position: {drone_position}, current cell: ({current_i}, {current_j})")

        fitness_scores = []

        for i, j in np.ndindex(map_data.shape):            
        
            ###### Distance #####
            x_cell = distance_between_cells*i - map_center_offset
            y_cell = distance_between_cells*j - map_center_offset
            distance = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)
            distance_value = distance/distance_norm

            ##### Trajectory average accomulate fitness #####
            trajectory_cells = self.get_cells_visited_in_trajectory(
                drone_altitude=drone_position[2],
                initial_cell=(current_i, current_j),
                final_cell=(i, j)
            )
            #### The sum of the fitness of the values in the trajectory devided bu the number of cells it will visit ####             
            average_trajectory_cells = sum([map_data[cell[0], cell[1]] for cell in trajectory_cells])/len(trajectory_cells) if trajectory_cells else 0.0
            trajectory_value = average_trajectory_cells/average_trajectory_accomulate_fitness_norm


            ##### Final fitness #####
            cell_fitness = trajectory_value  - distance_value
            fitness_scores.append((cell_fitness, (i, j)))

        return fitness_scores
    
    def two_cells_fitness(self, map_data: np.array, drone_position: Tuple[float, float, float],
                          distance_norm: float,
                          average_trajectory_accomulate_fitness_norm: float,
                          map_center_offset: float,
                          distance_between_cells: int,
                          another_drone_position: tuple,
                          distance_between_drone_norm: float) -> list:

        # 1. Get the lists (Keep this part the same)
        # If these methods are very slow, we can parallelize just these two lines later.
        current_drone_fitness_list = self.cells_fitness(map_data, drone_position, distance_norm, 
                                                        average_trajectory_accomulate_fitness_norm, 
                                                        map_center_offset, distance_between_cells)

        another_drone_fitness_list = self.cells_fitness(map_data, another_drone_position, distance_norm, 
                                                        average_trajectory_accomulate_fitness_norm, 
                                                        map_center_offset, distance_between_cells)

        # If lists are empty, return early to avoid errors
        if not current_drone_fitness_list or not another_drone_fitness_list:
            return []

        # 2. Convert to NumPy Arrays for speed
        # Structure: [fitness, cell_x, cell_y]
        # We split them into separate arrays for vectorization
        curr_data = np.array(current_drone_fitness_list, dtype=object)
        curr_fit = curr_data[:, 0].astype(float)
        curr_cells = np.array(curr_data[:, 1].tolist()) # Shape (N, 2)

        anoth_data = np.array(another_drone_fitness_list, dtype=object)
        anoth_fit = anoth_data[:, 0].astype(float)
        anoth_cells = np.array(anoth_data[:, 1].tolist()) # Shape (M, 2)

        # 3. Vectorized Coordinate Calculation
        # Apply the scaling math to the whole array at once
        # Shape: (N, 2) and (M, 2)
        curr_coords = (distance_between_cells * curr_cells) - map_center_offset
        anoth_coords = (distance_between_cells * anoth_cells) - map_center_offset

        # 4. Calculate Distance Matrix (Broadcasting)
        # We want distance between EVERY 'curr' and EVERY 'anoth'
        # Use broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diff = curr_coords[:, np.newaxis, :] - anoth_coords[np.newaxis, :, :]

        # Square, Sum, Sqrt (Euclidean distance)
        # Shape: (N, M)
        dists = np.sqrt(np.sum(diff**2, axis=2))

        # Normalize distances
        dist_penalty = dists / distance_between_drone_norm

        # 5. Calculate Combined Fitness Matrix
        # Shape: (N, 1) + (1, M) + (N, M) -> (N, M)
        total_fitness = curr_fit[:, np.newaxis] + anoth_fit[np.newaxis, :] + dist_penalty

        # 6. Formatting the output to match your original list structure
        # This part reconstructs the list of tuples: (fitness, cell1, cell2)

        # Get indices for N and M to reconstruct the pairs
        n_indices, m_indices = np.indices(total_fitness.shape)

        # Flatten everything
        flat_fitness = total_fitness.ravel()
        flat_n = n_indices.ravel()
        flat_m = m_indices.ravel()

        # Reconstruct the list comprehension using the original lists for cell references
        # (This is slightly slower than pure numpy but keeps your exact output format)
        combined_fitness_scores = [
            (flat_fitness[i], current_drone_fitness_list[flat_n[i]][1], another_drone_fitness_list[flat_m[i]][1])
            for i in range(len(flat_fitness))
        ]

        return combined_fitness_scores      

    
    def choose_one_cell(self, fitness_scores: list) -> Tuple[float, float]:
        if not fitness_scores:
            return None
        
        best_cell = max(fitness_scores, key=lambda x: x[0])
        #print(f"Chosen cell with fitness: {best_cell[0]}")
        # Return the coordinates
        return best_cell[1]

    def choose_two_cells(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        fitness_scores = max(fitness_scores, key=lambda x: x[0])
        #print(f"Chosen cells with fitness: {fitness_scores[0]}. So the positions are {fitness_scores[1]} and {fitness_scores[2]}")
        best_1 = (fitness_scores[1])
        best_2 = (fitness_scores[2])

        return [best_1, best_2]
        
