import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import RegularGridInterpolator


class SanityCheck():
    def __init__(self,fuzzy_parameters: np.array):

        self.create_fuzzy_controller(fuzzy_parameters)

    
    def get_interpolators(self):
        return self.interp_one_cell, self.interp_two_cells

    def create_fuzzy_controller(self, fuzzy_parameters: np.array):
        uncertainty_interval = fuzzy_parameters[0:3]
        distance_interval = fuzzy_parameters[3:6]
        one_cell_priority_interval = fuzzy_parameters[6:9]
        sum_of_priorities_interval = fuzzy_parameters[9:12]
        distance_between_targets_interval = fuzzy_parameters[12:15]
        two_cells_priority_interval = fuzzy_parameters[15:18]

        first_system_rules = fuzzy_parameters[18:27]
        second_system_rules = fuzzy_parameters[27:36]


        #############################
        #############################
        ###### FIRST SYSTEM #########
        #############################
        #############################
        uncertainty = ctrl.Antecedent(np.arange(0, 2, 0.02), 'uncertainty')
        distance = ctrl.Antecedent(np.arange(0, 30*10, 5.0), 'distance')
        one_cell_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'one_cell_priority', defuzzify_method = 'centroid')

        uncertainty['low'] = fuzz.trapmf(uncertainty.universe, [-1, 0, uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1]])
        uncertainty['medium'] = fuzz.trimf(uncertainty.universe, [uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]])
        uncertainty['high'] = fuzz.trapmf(uncertainty.universe, [uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2], 20.0, 21.0])
        
        distance['close'] = fuzz.trapmf(distance.universe, [-1, 0, distance_interval[0], distance_interval[0]+distance_interval[1]])
        distance['medium'] = fuzz.trimf(distance.universe, [distance_interval[0], distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2]])
        distance['far'] = fuzz.trapmf(distance.universe, [distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2], 300, 301])

        one_cell_priority['very_low'] = fuzz.trimf(one_cell_priority.universe, [-0.1, 0.0, one_cell_priority_interval[0]])
        one_cell_priority['low'] = fuzz.trimf(one_cell_priority.universe, [0.0, one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1]])
        one_cell_priority['medium'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]])
        one_cell_priority['high'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], 1.0])
        one_cell_priority['very_high'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], 1.0, 1.1])

        #####sanity checks
        uncertainty.view()
        plt.show() 
        distance.view()
        plt.show()
        one_cell_priority.view()
        plt.show()
        

        uncertainty_sets = ['low', 'medium', 'high']
        distance_sets = ['close', 'medium', 'far']
        one_cell_priority_sets = ['very_low', 'low', 'medium', 'high', 'very_high'] ### 0, 1, .., 4

        ### The gene will have also a vector for tuning the rules. This vector has size
        ### 9, because there are nine different combinations between the inputs. For each
        ### input combination it will have one of 5 possible outputs (very_low,...,very_high)
        ### consequently, the value of the vector is an integer from 0 to 4 (5 values). It's assumed that
        ### for each input combination there is an output in the system. 
        idx = 0
        first_system_active_rules = []
        for u in uncertainty_sets:
            for d in distance_sets:
                selected_priority = one_cell_priority_sets[int(first_system_rules[idx])]

                rule = ctrl.Rule(antecedent=(uncertainty[u] & distance[d]), 
                                consequent=one_cell_priority[selected_priority])
                

                print(f"Rule: IF uncertainty is {u} AND distance is {d} THEN one_cell_priority is {selected_priority}")
                first_system_active_rules.append(rule)
                idx += 1
        
        one_cell_fuzzy = ctrl.ControlSystem(first_system_active_rules)
        self.one_cell_priority = ctrl.ControlSystemSimulation(one_cell_fuzzy)


        
        #############################
        #############################
        ###### SECOND SYSTEM ########
        #############################
        #############################
        sum_priorities = ctrl.Antecedent(np.arange(0, 2.0, 0.02), 'sum_priorities')
        distance_between_targets = ctrl.Antecedent(np.arange(0, 30*10, 5.0), 'distance_between_targets')
        pair_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'pair_priority', defuzzify_method = 'centroid')

        sum_priorities['low'] = fuzz.trapmf(sum_priorities.universe, [-0.1, 0.0, sum_of_priorities_interval[0], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]])
        sum_priorities['medium'] = fuzz.trimf(sum_priorities.universe, [sum_of_priorities_interval[0], sum_of_priorities_interval[0]+sum_of_priorities_interval[1], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]])
        sum_priorities['high'] = fuzz.trapmf(sum_priorities.universe, [sum_of_priorities_interval[0]+sum_of_priorities_interval[1], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2], 2.0, 2.1])

        distance_between_targets['close'] = fuzz.trapmf(distance_between_targets.universe, [-1, 0, distance_between_targets_interval[0], distance_between_targets_interval[0]+distance_between_targets_interval[1]])
        distance_between_targets['medium'] = fuzz.trimf(distance_between_targets.universe, [distance_between_targets_interval[0], distance_between_targets_interval[0]+distance_between_targets_interval[1], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]])
        distance_between_targets['far'] = fuzz.trapmf(distance_between_targets.universe, [distance_between_targets_interval[0]+distance_between_targets_interval[1], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2], 300, 301])

        pair_priority['very_low'] = fuzz.trimf(pair_priority.universe, [-0.1, 0.0, two_cells_priority_interval[0]])
        pair_priority['low'] = fuzz.trimf(pair_priority.universe, [0.0, two_cells_priority_interval[0], two_cells_priority_interval[0]+two_cells_priority_interval[1]])
        pair_priority['medium'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0], two_cells_priority_interval[0]+two_cells_priority_interval[1], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]])
        pair_priority['high'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2], 1.0])
        pair_priority['very_high'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2], 1.0, 1.1])
        
        
        sum_priorities.view()
        plt.show() 
        distance_between_targets.view()
        plt.show()
        pair_priority.view()
        plt.show()
        ### Fuzzy Rules
        # Sets
        sum_of_uncertainty_sets = ['low', 'medium', 'high']
        distance_between_targets_sets = ['close', 'medium', 'far']
        two_cell_priority_sets = ['very_low', 'low', 'medium', 'high', 'very_high'] ### 0, 1, .., 4

        idx = 0
        second_system_active_rules = []
        for u in sum_of_uncertainty_sets:
            for d in distance_between_targets_sets:
                selected_priority = two_cell_priority_sets[int(second_system_rules[idx])]

                rule = ctrl.Rule(antecedent=(sum_priorities[u] & distance_between_targets[d]), 
                                consequent=pair_priority[selected_priority])
                second_system_active_rules.append(rule)
                idx += 1

                print(f"Rule: IF sum_priorities is {u} AND distance_between_targets is {d} THEN pair_priority is {selected_priority}")
        
        two_cells_fuzzy = ctrl.ControlSystem(second_system_active_rules)
        self.two_cells_priority = ctrl.ControlSystemSimulation(two_cells_fuzzy)



def main():
    individual = [np.float64(0.08879789658449988), np.float64(0.2990584261575781), np.float64(0.32384850500067697),
            np.float64(59.67149923152216), np.float64(74.61556171321105), np.float64(87.84193135295521),
            np.float64(0.2771856390455823), np.float64(0.2787134317315393), np.float64(0.1638125494958381),
            np.float64(0.673782021688156), np.float64(0.6329466169294924), np.float64(0.5591202075074384),
            np.float64(50.54463178241406), np.float64(42.27566047833311), np.float64(44.5656822753396),
            np.float64(0.32012394689951496), np.float64(0.2040906672593554), np.float64(0.280814060613372),
            np.int64(0), np.int64(2), np.int64(1),
            np.int64(0), np.int64(4), np.int64(1),
            np.int64(3), np.int64(4), np.int64(3),
            np.int64(1), np.int64(0), np.int64(2),
            np.int64(2), np.int64(3), np.int64(3),
            np.int64(4), np.int64(3), np.int64(4)] 
    evaluator = SanityCheck(fuzzy_parameters=individual)
    
if __name__ == "__main__":
    main()