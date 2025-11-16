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
        

    def cells_fitness(self, map_data: np.array, drone_position: Tuple[float, float, float], distance_norm: float, uncertainty_norm: float, trajectory_accomulate_fitness_norm: float, map_center_offset: float, distance_between_cells: int) -> list:

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

            ##### Trajectory accomulate fitness #####
            trajectory_cells = self.get_cells_visited_in_trajectory(
                drone_altitude=drone_position[2],
                initial_cell=(current_i, current_j),
                final_cell=(i, j)
            )
            #print(f"Trajectory from ({current_i}, {current_j}) to ({i}, {j}) covers cells: {trajectory_cells}")
            trajectory_value = sum([map_data[cell[0], cell[1]] for cell in trajectory_cells])/trajectory_accomulate_fitness_norm
            #print(f"Trajectory accomulate fitness value for cell ({i}, {j}): {trajectory_value}")
            ##### Uncertainty #####
            uncertainty_value = map_data[i, j]/uncertainty_norm

            ##### Final fitness #####
            cell_fitness = uncertainty_value + trajectory_value  - distance_value
            fitness_scores.append((cell_fitness, (i, j)))

        return fitness_scores
    
    def two_cells_fitness(self, map_data: np.array, drone_position: Tuple[float, float, float], distance_norm: float, uncertainty_norm: float, trajectory_accomulate_fitness_norm: float, map_center_offset: float, distance_between_cells: int, another_drone_position:tuple, distance_between_drone_norm:float) -> list:
        combined_fitness_scores = []

        current_drone_fitness_list = self.cells_fitness(map_data,
                                                        drone_position,
                                                        distance_norm,
                                                        uncertainty_norm,
                                                        trajectory_accomulate_fitness_norm,
                                                        map_center_offset,
                                                        distance_between_cells)

        another_drone_fitness_list = self.cells_fitness(map_data,
                                                        another_drone_position,
                                                        distance_norm,
                                                        uncertainty_norm,
                                                        trajectory_accomulate_fitness_norm,
                                                        map_center_offset,
                                                        distance_between_cells)
        
        for current_drone_index in range(len(current_drone_fitness_list)):
            current_fitness, current_cell = current_drone_fitness_list[current_drone_index]
            
            for another_drone_index in range(len(another_drone_fitness_list)):
                another_fitness, another_cell = another_drone_fitness_list[another_drone_index]

                #### Distance between drones penalty ####
                x_cell = distance_between_cells*current_cell[0] - map_center_offset
                y_cell = distance_between_cells*current_cell[1] - map_center_offset

                x_another_cell = distance_between_cells*another_cell[0] - map_center_offset
                y_another_cell = distance_between_cells*another_cell[1] - map_center_offset

                distance_between_targets = np.sqrt((x_cell - x_another_cell) ** 2 + (y_cell - y_another_cell) ** 2)
                distance_between_drones_value = distance_between_targets/distance_between_drone_norm

                # Updating fitness with distance between drones penalty
                combined_fitness = current_fitness + another_fitness + distance_between_drones_value
                combined_fitness_scores.append((combined_fitness, current_cell, another_cell))
        
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
        
