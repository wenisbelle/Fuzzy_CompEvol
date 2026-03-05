import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import RegularGridInterpolator

class FuzzyLookupTable:
    def __init__(self,fuzzy_parameters: np.array):

        self.create_fuzzy_controller(fuzzy_parameters)
        print(f"Starting the lookup generation...")
        # 2. GENERATE LOOKUP TABLES (The Optimization)
        self.interp_one_cell = self._precompute_one_cell_surface()
        self.interp_two_cells = self._precompute_two_cells_surface()
        print(f"Finished the lookup generation...")
    
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
                first_system_active_rules.append(rule)
                idx += 1
        
        one_cell_fuzzy = ctrl.ControlSystem(first_system_active_rules)
        self.one_cell_priority = ctrl.ControlSystemSimulation(one_cell_fuzzy)


        
        #############################
        #############################
        ###### SECOND SYSTEM ########
        #############################
        #############################
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
        
        two_cells_fuzzy = ctrl.ControlSystem(second_system_active_rules)
        self.two_cells_priority = ctrl.ControlSystemSimulation(two_cells_fuzzy)


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

