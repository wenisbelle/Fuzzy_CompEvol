import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyEvaluator:
    def __init__(self, map_width: int, map_height: int, camera_angle:float, fuzzy_parameters: np.array, distance_between_cells: int =10):
        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells
        
        uncertainty_interval = fuzzy_parameters[0:3]
        distance_interval = fuzzy_parameters[3:6]
        individual_cell_uncertainty_interval = fuzzy_parameters[6:9]
        one_cell_priority_interval = fuzzy_parameters[9:12]


        ## Creating the fuzzy system without the ga optimization
        ### First system
        self.uncertainty = ctrl.Antecedent(np.arange(0, 20, 0.01), 'uncertainty')
        self.distance = ctrl.Antecedent(np.arange(0, 150, 1.0), 'distance')
        self.individual_cell_uncertainty = ctrl.Antecedent(np.arange(0, 2.0, 0.01), 'individual_cell_uncertainty')
        self.one_cell_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'one_cell_priority', defuzzify_method = 'centroid')

        self.uncertainty['low'] = fuzz.trapmf(self.uncertainty.universe, [-1, 0, uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1]])
        self.uncertainty['medium'] = fuzz.trimf(self.uncertainty.universe, [uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]])
        self.uncertainty['high'] = fuzz.trapmf(self.uncertainty.universe, [uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2], 20.0, 21.0])
        
        self.distance['close'] = fuzz.trapmf(self.distance.universe, [-1, 0, distance_interval[0], distance_interval[0]+distance_interval[1]])
        self.distance['medium'] = fuzz.trimf(self.distance.universe, [distance_interval[0], distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2]])
        self.distance['far'] = fuzz.trapmf(self.distance.universe, [distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2], 150, 151])

        self.individual_cell_uncertainty['low'] = fuzz.trapmf(self.individual_cell_uncertainty.universe, [-0.1, 0.0, individual_cell_uncertainty_interval[0], individual_cell_uncertainty_interval[0]+individual_cell_uncertainty_interval[1]])
        self.individual_cell_uncertainty['medium'] = fuzz.trimf(self.individual_cell_uncertainty.universe, [individual_cell_uncertainty_interval[0], individual_cell_uncertainty_interval[0]+individual_cell_uncertainty_interval[1], individual_cell_uncertainty_interval[0]+individual_cell_uncertainty_interval[1]+individual_cell_uncertainty_interval[2]])
        self.individual_cell_uncertainty['high'] = fuzz.trapmf(self.individual_cell_uncertainty.universe, [individual_cell_uncertainty_interval[0]+individual_cell_uncertainty_interval[1], individual_cell_uncertainty_interval[0]+individual_cell_uncertainty_interval[1]+individual_cell_uncertainty_interval[2], 2.0, 2.1])
        
        self.one_cell_priority['very_low'] = fuzz.trimf(self.one_cell_priority.universe, [-0.1, 0.0, one_cell_priority_interval[0]])
        self.one_cell_priority['low'] = fuzz.trimf(self.one_cell_priority.universe, [0.0, one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1]])
        self.one_cell_priority['medium'] = fuzz.trimf(self.one_cell_priority.universe, [one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]])
        self.one_cell_priority['high'] = fuzz.trimf(self.one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], 1.0])
        self.one_cell_priority['very_high'] = fuzz.trimf(self.one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], 1.0, 1.1])

        #####sanity checks
        #self.uncertainty.view()
        #plt.show() 
        #self.distance.view()
        #plt.show()
        #self.individual_cell_uncertainty.view()
        #plt.show()
        #self.one_cell_priority.view()
        #plt.show()
        # 

        #### Rules
        FS1_rule1 = ctrl.Rule(self.uncertainty['high'] & self.distance['close'], self.one_cell_priority['very_high'])
        FS1_rule2 = ctrl.Rule(self.uncertainty['high'] & self.distance['medium'], self.one_cell_priority['very_high'])
        FS1_rule3 = ctrl.Rule(self.uncertainty['high'] & self.distance['far'], self.one_cell_priority['medium'])
        
        FS1_rule4 = ctrl.Rule(self.uncertainty['medium'] & self.distance['close'] & self.individual_cell_uncertainty['high'], self.one_cell_priority['very_high'])
        FS1_rule5 = ctrl.Rule(self.uncertainty['medium'] & self.distance['close'] & self.individual_cell_uncertainty['medium'], self.one_cell_priority['medium'])
        FS1_rule6 = ctrl.Rule(self.uncertainty['medium'] & self.distance['close'] & self.individual_cell_uncertainty['low'], self.one_cell_priority['very_low'])
        
        FS1_rule7 = ctrl.Rule(self.uncertainty['medium'] & self.distance['medium'] & self.individual_cell_uncertainty['high'], self.one_cell_priority['high'])
        FS1_rule8 = ctrl.Rule(self.uncertainty['medium'] & self.distance['medium'] & self.individual_cell_uncertainty['medium'], self.one_cell_priority['medium'])
        FS1_rule9 = ctrl.Rule(self.uncertainty['medium'] & self.distance['medium'] & self.individual_cell_uncertainty['low'], self.one_cell_priority['very_low'])

        FS1_rule10 = ctrl.Rule(self.uncertainty['medium'] & self.distance['far'] & self.individual_cell_uncertainty['high'], self.one_cell_priority['medium'])
        FS1_rule11 = ctrl.Rule(self.uncertainty['medium'] & self.distance['far'] & self.individual_cell_uncertainty['medium'], self.one_cell_priority['low'])
        FS1_rule12 = ctrl.Rule(self.uncertainty['medium'] & self.distance['far'] & self.individual_cell_uncertainty['low'], self.one_cell_priority['very_low'])

        FS1_rule13 = ctrl.Rule(self.uncertainty['low'] & self.distance['close'] & self.individual_cell_uncertainty['high'], self.one_cell_priority['medium'])
        FS1_rule14 = ctrl.Rule(self.uncertainty['low'] & self.distance['close'] & self.individual_cell_uncertainty['medium'], self.one_cell_priority['low'])
        FS1_rule15 = ctrl.Rule(self.uncertainty['low'] & self.distance['close'] & self.individual_cell_uncertainty['low'], self.one_cell_priority['very_low'])

        FS1_rule16 = ctrl.Rule(self.uncertainty['low'] & self.distance['medium'] & self.individual_cell_uncertainty['high'], self.one_cell_priority['low'])
        FS1_rule17 = ctrl.Rule(self.uncertainty['low'] & self.distance['medium'] & self.individual_cell_uncertainty['medium'], self.one_cell_priority['very_low'])
        FS1_rule18 = ctrl.Rule(self.uncertainty['low'] & self.distance['medium'] & self.individual_cell_uncertainty['low'], self.one_cell_priority['very_low'])

        FS1_rule19 = ctrl.Rule(self.uncertainty['low'] & self.distance['far'], self.one_cell_priority['very_low'])

        one_cell_fuzzy = ctrl.ControlSystem([FS1_rule1, FS1_rule2, FS1_rule3, FS1_rule4, FS1_rule5, FS1_rule6, FS1_rule7, FS1_rule8, FS1_rule9, 
                                            FS1_rule10, FS1_rule11, FS1_rule12, FS1_rule13, FS1_rule14, FS1_rule15, FS1_rule16, FS1_rule17, FS1_rule18, FS1_rule19])
        self.one_cell_priority = ctrl.ControlSystemSimulation(one_cell_fuzzy)

        #check values
        #one_cell_priority.input['uncertainty'] = 200
        #one_cell_priority.input['distance'] = 25
        #one_cell_priority.compute()
        #print(f"Test output: {one_cell_priority.output['one_cell_priority']}")

          
        self.sum_priorities = ctrl.Antecedent(np.arange(0, 2.0, 0.01), 'sum_priorities')
        self.distance_between_targets = ctrl.Antecedent(np.arange(0, 150, 1.0), 'distance_between_targets')
        self.pair_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'pair_priority', defuzzify_method = 'centroid')

        self.sum_priorities['low'] = fuzz.trapmf(self.sum_priorities.universe, [-0.1, 0.0, 0.5, 1.0])
        self.sum_priorities['medium'] = fuzz.trimf(self.sum_priorities.universe, [0.5, 1.0, 1.5])
        self.sum_priorities['high'] = fuzz.trapmf(self.sum_priorities.universe, [1.0, 1.5, 2.0, 2.1])

        self.distance_between_targets['close'] = fuzz.trapmf(self.distance_between_targets.universe, [-1, 0, distance_interval[0], distance_interval[0]+distance_interval[1]])
        self.distance_between_targets['medium'] = fuzz.trimf(self.distance_between_targets.universe, [distance_interval[0], distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2]])
        self.distance_between_targets['far'] = fuzz.trapmf(self.distance_between_targets.universe, [distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2], 150, 151])

        self.pair_priority['very_low'] = fuzz.trimf(self.pair_priority.universe, [-0.1, 0.0, 0.25])
        self.pair_priority['low'] = fuzz.trimf(self.pair_priority.universe, [0.0, 0.25, 0.50])
        self.pair_priority['medium'] = fuzz.trimf(self.pair_priority.universe, [0.25, 0.50, 0.75])
        self.pair_priority['high'] = fuzz.trimf(self.pair_priority.universe, [0.50, 0.75, 1.0])
        self.pair_priority['very_high'] = fuzz.trimf(self.pair_priority.universe, [0.75, 1.0, 1.1])

        #####sanity checks
        #sum_priorities.view()
        #plt.show()
        #distance_between_targets.view()
        #plt.show()   
        #pair_priority.view()
        #plt.show() 

        ### Fuzzy Rules
        FS2_rule1 = ctrl.Rule(self.sum_priorities['high'] & self.distance_between_targets['far'], self.pair_priority['very_high'])
        FS2_rule2 = ctrl.Rule(self.sum_priorities['high'] & self.distance_between_targets['medium'], self.pair_priority['very_high'])
        FS2_rule3 = ctrl.Rule(self.sum_priorities['high'] & self.distance_between_targets['close'], self.pair_priority['very_low'])
        FS2_rule4 = ctrl.Rule(self.sum_priorities['medium'] & self.distance_between_targets['far'], self.pair_priority['high'])
        FS2_rule5 = ctrl.Rule(self.sum_priorities['medium'] & self.distance_between_targets['medium'], self.pair_priority['medium'])
        FS2_rule6 = ctrl.Rule(self.sum_priorities['medium'] & self.distance_between_targets['close'], self.pair_priority['very_low'])
        FS2_rule7 = ctrl.Rule(self.sum_priorities['low'] & self.distance_between_targets['far'], self.pair_priority['low'])
        FS2_rule8 = ctrl.Rule(self.sum_priorities['low'] & self.distance_between_targets['medium'], self.pair_priority['very_low'])
        FS2_rule9 = ctrl.Rule(self.sum_priorities['low'] & self.distance_between_targets['close'], self.pair_priority['very_low'])

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

    def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float, distance_between_cells: int) -> list:

        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        #### Getting the current index of the drone in the map ####
        current_i = int((drone_x + map_center_offset)/distance_between_cells)
        current_j = int((drone_y + map_center_offset)/distance_between_cells)

        priority_scores = []

        for i, j in np.ndindex(map_data.shape):            
        
            ###### Distance #####
            x_cell = distance_between_cells*i - map_center_offset
            y_cell = distance_between_cells*j - map_center_offset
            distance = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)
            
            ##### Trajectory accomulate fitness #####
            trajectory_cells = self.get_cells_visited_in_trajectory(
                drone_altitude=drone_position[2],
                initial_cell=(current_i, current_j),
                final_cell=(i, j)
            )
            trajectory_value = sum([map_data[cell[0], cell[1]] for cell in trajectory_cells])
            if trajectory_value > 20:
                trajectory_value = 20

            ##### Individual cell uncertainty #####
            individual_uncertainty = map_data[i, j]
            if individual_uncertainty > 2.0:
                individual_uncertainty = 2.0
            
            ##### Priority #####
            self.one_cell_priority.input['uncertainty'] = trajectory_value
            self.one_cell_priority.input['distance'] = distance
            self.one_cell_priority.input['individual_cell_uncertainty'] = individual_uncertainty  
            self.one_cell_priority.compute()
            priority = self.one_cell_priority.output['one_cell_priority']            
            ### Append the result
            priority_scores.append((priority, (i, j)))

        return priority_scores

    def both_cells_priority(self, map_data: np.array, first_drone_position: Tuple[float, float, float], second_drone_position: Tuple[float, float, float], map_center_offset: float, distance_between_cells: int) -> list:
        combined_priority_scores = []
        first_drone_priority_score_list = self.cells_priority(map_data, first_drone_position, map_center_offset, distance_between_cells)
        second_drone_priority_score_list = self.cells_priority(map_data, second_drone_position, map_center_offset, distance_between_cells)

        for first_drone_index in range(len(first_drone_priority_score_list)):
            first_priority, first_cell = first_drone_priority_score_list[first_drone_index]

            for second_drone_index in range(len(second_drone_priority_score_list)):
                second_priority, second_cell = second_drone_priority_score_list[second_drone_index]

            #### Distance between drones penalty ####
            x_first_cell = distance_between_cells*first_cell[0] - map_center_offset
            y_first_cell = distance_between_cells*first_cell[1] - map_center_offset
            x_second_cell = distance_between_cells*second_cell[0] - map_center_offset
            y_second_cell = distance_between_cells*second_cell[1] - map_center_offset

            distance_between_targets = np.sqrt((x_first_cell-x_second_cell)**2 + (y_first_cell-y_second_cell)**2)

            self.two_cells_priority.input['sum_priorities'] = first_priority + second_priority
            self.two_cells_priority.input['distance_between_targets'] = distance_between_targets
            self.two_cells_priority.compute()
            combined_priority = self.two_cells_priority.output['pair_priority']
            combined_priority_scores.append((combined_priority, first_cell, second_cell))
            #print(f"Pair ({first_cell}, {second_cell}): Combined Priority={combined_priority}")

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