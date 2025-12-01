import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import RegularGridInterpolator

class FuzzyEvaluator:
    def __init__(self, map_width: int, map_height: int, camera_angle:float, fuzzy_parameters: np.array, distance_between_cells: int =10):
        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells

        self.create_fuzzy_controller(fuzzy_parameters)

        # 2. GENERATE LOOKUP TABLES (The Optimization)
        print("Pre-computing fuzzy control surfaces... (this happens once)")
        self.interp_one_cell = self._precompute_one_cell_surface()
        self.interp_two_cells = self._precompute_two_cells_surface()
        print("Pre-computation complete.")       
        

    def create_fuzzy_controller(self, fuzzy_parameters: np.array):
        uncertainty_interval = fuzzy_parameters[0:3]
        distance_interval = fuzzy_parameters[3:6]
        one_cell_priority_interval = fuzzy_parameters[6:9]
        sum_of_priorities_interval = fuzzy_parameters[9:12]
        distance_between_targets_interval = fuzzy_parameters[12:15]
        two_cells_priority_interval = fuzzy_parameters[15:18]


        ### First system
        uncertainty = ctrl.Antecedent(np.arange(0, 2, 0.05), 'uncertainty')
        distance = ctrl.Antecedent(np.arange(0, 150, 5.0), 'distance')
        one_cell_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'one_cell_priority', defuzzify_method = 'centroid')

        uncertainty['low'] = fuzz.trapmf(uncertainty.universe, [-1, 0, uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1]])
        uncertainty['medium'] = fuzz.trimf(uncertainty.universe, [uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]])
        uncertainty['high'] = fuzz.trapmf(uncertainty.universe, [uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2], 20.0, 21.0])
        
        distance['close'] = fuzz.trapmf(distance.universe, [-1, 0, distance_interval[0], distance_interval[0]+distance_interval[1]])
        distance['medium'] = fuzz.trimf(distance.universe, [distance_interval[0], distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2]])
        distance['far'] = fuzz.trapmf(distance.universe, [distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2], 150, 151])

        one_cell_priority['very_low'] = fuzz.trimf(one_cell_priority.universe, [-0.1, 0.0, one_cell_priority_interval[0]])
        one_cell_priority['low'] = fuzz.trimf(one_cell_priority.universe, [0.0, one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1]])
        one_cell_priority['medium'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]])
        one_cell_priority['high'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], 1.0])
        one_cell_priority['very_high'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], 1.0, 1.1])

        #####sanity checks
        #uncertainty.view()
        #plt.show() 
        #distance.view()
        #plt.show()
        #individual_cell_uncertainty.view()
        #plt.show()
        #one_cell_priority.view()
        #plt.show()
        # 

        #### Rules
        FS1_rule1 = ctrl.Rule(uncertainty['high'] & distance['close'], one_cell_priority['very_high'])
        FS1_rule2 = ctrl.Rule(uncertainty['high'] & distance['medium'], one_cell_priority['high'])
        FS1_rule3 = ctrl.Rule(uncertainty['high'] & distance['far'], one_cell_priority['medium'])
        
        FS1_rule4 = ctrl.Rule(uncertainty['medium'] & distance['close'], one_cell_priority['high'])
        FS1_rule5 = ctrl.Rule(uncertainty['medium'] & distance['medium'], one_cell_priority['medium'])
        FS1_rule6 = ctrl.Rule(uncertainty['medium'] & distance['far'], one_cell_priority['low'])        

        FS1_rule7 = ctrl.Rule(uncertainty['low'] & distance['close'], one_cell_priority['medium'])
        FS1_rule8 = ctrl.Rule(uncertainty['low'] & distance['medium'], one_cell_priority['low'])
        FS1_rule9 = ctrl.Rule(uncertainty['low'] & distance['far'], one_cell_priority['very_low'])

        one_cell_fuzzy = ctrl.ControlSystem([FS1_rule1, FS1_rule2, FS1_rule3, FS1_rule4, FS1_rule5, FS1_rule6, FS1_rule7, FS1_rule8, FS1_rule9])
        self.one_cell_priority = ctrl.ControlSystemSimulation(one_cell_fuzzy)

        #check values
        #one_cell_priority.input['uncertainty'] = 200
        #one_cell_priority.input['distance'] = 25
        #one_cell_priority.compute()
        #print(f"Test output: {one_cell_priority.output['one_cell_priority']}")

          
        sum_priorities = ctrl.Antecedent(np.arange(0, 2.0, 0.01), 'sum_priorities')
        distance_between_targets = ctrl.Antecedent(np.arange(0, 150, 1.0), 'distance_between_targets')
        pair_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'pair_priority', defuzzify_method = 'centroid')

        sum_priorities['low'] = fuzz.trapmf(sum_priorities.universe, [-0.1, 0.0, sum_of_priorities_interval[0], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]])
        sum_priorities['medium'] = fuzz.trimf(sum_priorities.universe, [sum_of_priorities_interval[0], sum_of_priorities_interval[0]+sum_of_priorities_interval[1], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]])
        sum_priorities['high'] = fuzz.trapmf(sum_priorities.universe, [sum_of_priorities_interval[0]+sum_of_priorities_interval[1], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2], 2.0, 2.1])

        distance_between_targets['close'] = fuzz.trapmf(distance_between_targets.universe, [-1, 0, distance_between_targets_interval[0], distance_between_targets_interval[0]+distance_between_targets_interval[1]])
        distance_between_targets['medium'] = fuzz.trimf(distance_between_targets.universe, [distance_between_targets_interval[0], distance_between_targets_interval[0]+distance_between_targets_interval[1], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]])
        distance_between_targets['far'] = fuzz.trapmf(distance_between_targets.universe, [distance_between_targets_interval[0]+distance_between_targets_interval[1], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2], 150, 151])

        pair_priority['very_low'] = fuzz.trimf(pair_priority.universe, [-0.1, 0.0, two_cells_priority_interval[0]])
        pair_priority['low'] = fuzz.trimf(pair_priority.universe, [0.0, two_cells_priority_interval[0], two_cells_priority_interval[0]+two_cells_priority_interval[1]])
        pair_priority['medium'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0], two_cells_priority_interval[0]+two_cells_priority_interval[1], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]])
        pair_priority['high'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2], 1.0])
        pair_priority['very_high'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2], 1.0, 1.1])

        #####sanity checks
        #sum_priorities.view()
        #plt.show()
        #distance_between_targets.view()
        #plt.show()   
        #pair_priority.view()
        #plt.show() 

        ### Fuzzy Rules
        FS2_rule1 = ctrl.Rule(sum_priorities['high'] & distance_between_targets['far'], pair_priority['very_high'])
        FS2_rule2 = ctrl.Rule(sum_priorities['high'] & distance_between_targets['medium'], pair_priority['high'])
        FS2_rule3 = ctrl.Rule(sum_priorities['high'] & distance_between_targets['close'], pair_priority['low'])
        FS2_rule4 = ctrl.Rule(sum_priorities['medium'] & distance_between_targets['far'], pair_priority['high'])
        FS2_rule5 = ctrl.Rule(sum_priorities['medium'] & distance_between_targets['medium'], pair_priority['medium'])
        FS2_rule6 = ctrl.Rule(sum_priorities['medium'] & distance_between_targets['close'], pair_priority['low'])
        FS2_rule7 = ctrl.Rule(sum_priorities['low'] & distance_between_targets['far'], pair_priority['medium'])
        FS2_rule8 = ctrl.Rule(sum_priorities['low'] & distance_between_targets['medium'], pair_priority['low'])
        FS2_rule9 = ctrl.Rule(sum_priorities['low'] & distance_between_targets['close'], pair_priority['very_low'])

        two_cells_fuzzy = ctrl.ControlSystem([FS2_rule1, FS2_rule2, FS2_rule3, FS2_rule4, FS2_rule5, FS2_rule6, FS2_rule7, FS2_rule8, FS2_rule9])
        self.two_cells_priority = ctrl.ControlSystemSimulation(two_cells_fuzzy)


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
        
    def _precompute_one_cell_surface(self):
        """Generates a 3D Lookup Table for the first fuzzy system."""
        u_range = np.linspace(0, 2, 40) 
        d_range = np.linspace(0, 150, 30)
        
        # Create a grid
        output_surface = np.zeros((len(u_range), len(d_range)))

        # Fill the grid (This part is slow, but only runs ONCE at startup)
        for i, u_val in enumerate(u_range):
            for j, d_val in enumerate(d_range):
                self.one_cell_priority.input['uncertainty'] = u_val
                self.one_cell_priority.input['distance'] = d_val
                try:
                    self.one_cell_priority.compute()
                    output_surface[i, j] = self.one_cell_priority.output['one_cell_priority']
                except (KeyError, ValueError):
                    output_surface[i, j] = 0.0
        
        # Create the interpolator function
        return RegularGridInterpolator((u_range, d_range), output_surface, bounds_error=False, fill_value=None)
    
    def _precompute_two_cells_surface(self):
        """Generates a 2D Lookup Table for the second fuzzy system."""
        s_range = np.linspace(0, 2.0, 40)
        d_range = np.linspace(0, 150, 30)
        
        output_surface = np.zeros((len(s_range), len(d_range)))

        for i, s_val in enumerate(s_range):
            for j, d_val in enumerate(d_range):
                self.two_cells_priority.input['sum_priorities'] = s_val
                self.two_cells_priority.input['distance_between_targets'] = d_val
                try:
                    self.two_cells_priority.compute()
                    output_surface[i, j] = self.two_cells_priority.output['pair_priority']
                except (KeyError, ValueError):
                    # If inputs fall into a gap where no rules fire, 
                    # or the area is zero, default the priority to 0.
                    output_surface[i, j] = 0.0

        return RegularGridInterpolator((s_range, d_range), output_surface, bounds_error=False, fill_value=None)

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

        # Iterate map
        # Note: If map is huge, we should vectorize distance calculation too.
        rows, cols = map_data.shape
        for i in range(rows):
            for j in range(cols):
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
    
##### For sanaty checks purposes only #####
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