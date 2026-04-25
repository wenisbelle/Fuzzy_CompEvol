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
        # 2. GENERATE LOOKUP TABLES (The Optimization)
        self.interp_one_cell = self._precompute_one_cell_surface()
        self.interp_two_cells = self._precompute_two_cells_surface()
    
    def get_interpolators(self):
        return self.interp_one_cell, self.interp_two_cells

    def create_fuzzy_controller(self, fuzzy_parameters: np.array):
        uncertainty_interval = fuzzy_parameters[0:5]
        distance_interval = fuzzy_parameters[5:10]
        one_cell_priority_interval = fuzzy_parameters[10:15]
        sum_of_priorities_interval = fuzzy_parameters[15:20]
        distance_between_targets_interval = fuzzy_parameters[20:25]
        two_cells_priority_interval = fuzzy_parameters[25:30]

        first_system_rules = fuzzy_parameters[30:55]
        second_system_rules = fuzzy_parameters[55:80]


        #############################
        #############################
        ###### FIRST SYSTEM #########
        #############################
        #############################
        uncertainty = ctrl.Antecedent(np.arange(0, 2, 0.02), 'uncertainty')
        distance = ctrl.Antecedent(np.arange(0, 30*10, 5.0), 'distance')
        one_cell_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'one_cell_priority', defuzzify_method = 'centroid')

        uncertainty['very_low'] = fuzz.trapmf(uncertainty.universe, [-1, 0, uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1]])
        uncertainty['low'] = fuzz.trimf(uncertainty.universe, [uncertainty_interval[0], uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]])
        uncertainty['medium'] = fuzz.trimf(uncertainty.universe, [uncertainty_interval[0]+uncertainty_interval[1], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]+uncertainty_interval[3]])
        uncertainty['high'] = fuzz.trimf(uncertainty.universe, [uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]+uncertainty_interval[3], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]+uncertainty_interval[3]+uncertainty_interval[4]])      
        uncertainty['very_high'] = fuzz.trapmf(uncertainty.universe, [uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]+uncertainty_interval[3], uncertainty_interval[0]+uncertainty_interval[1]+uncertainty_interval[2]+uncertainty_interval[3]+uncertainty_interval[4], 2.0, 2.1])
        
        distance['very_close'] = fuzz.trapmf(distance.universe, [-1, 0, distance_interval[0], distance_interval[0]+distance_interval[1]])
        distance['close'] = fuzz.trimf(distance.universe, [distance_interval[0], distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2]])
        distance['medium'] = fuzz.trimf(distance.universe, [distance_interval[0]+distance_interval[1], distance_interval[0]+distance_interval[1]+distance_interval[2], distance_interval[0]+distance_interval[1]+distance_interval[2]+distance_interval[3]])
        distance['far'] = fuzz.trimf(distance.universe, [distance_interval[0]+distance_interval[1]+distance_interval[2], distance_interval[0]+distance_interval[1]+distance_interval[2]+distance_interval[3], distance_interval[0]+distance_interval[1]+distance_interval[2]+distance_interval[3]+distance_interval[4]])   
        distance['very_far'] = fuzz.trapmf(distance.universe, [distance_interval[0]+distance_interval[1]+distance_interval[2]+distance_interval[3], distance_interval[0]+distance_interval[1]+distance_interval[2]+distance_interval[3]+distance_interval[4], 300.0, 300.1])
        
        one_cell_priority['very_low'] = fuzz.trapmf(one_cell_priority.universe, [-1, 0, one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1]])
        one_cell_priority['low'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0], one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]])
        one_cell_priority['medium'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]+one_cell_priority_interval[3]])
        one_cell_priority['high'] = fuzz.trimf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]+one_cell_priority_interval[3], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]+one_cell_priority_interval[3]+one_cell_priority_interval[4]])   
        one_cell_priority['very_high'] = fuzz.trapmf(one_cell_priority.universe, [one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]+one_cell_priority_interval[3], one_cell_priority_interval[0]+one_cell_priority_interval[1]+one_cell_priority_interval[2]+one_cell_priority_interval[3]+one_cell_priority_interval[4], 300.0, 300.1])


        uncertainty_sets = ['very_low', 'low' , 'medium', 'high', 'very_high']
        distance_sets = ['very_close', 'close', 'medium', 'far', 'very_far']
        one_cell_priority_sets = ['very_low', 'low', 'medium', 'high', 'very_high'] ### 0, 1, .., 4

        ### The gene will have also a vector for tuning the rules. This vector has size
        ### 25, because there are nine different combinations between the inputs. For each
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
        sum_priorities = ctrl.Antecedent(np.arange(0, 3.0, 0.02), 'sum_priorities')
        distance_between_targets = ctrl.Antecedent(np.arange(0, 40*10, 5.0), 'distance_between_targets')
        pair_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'pair_priority', defuzzify_method='centroid')

        sum_priorities['very_low'] = fuzz.trapmf(sum_priorities.universe, [-1, 0, sum_of_priorities_interval[0], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]])
        sum_priorities['low'] = fuzz.trimf(sum_priorities.universe, [sum_of_priorities_interval[0], sum_of_priorities_interval[0]+sum_of_priorities_interval[1], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]])
        sum_priorities['medium'] = fuzz.trimf(sum_priorities.universe, [sum_of_priorities_interval[0]+sum_of_priorities_interval[1], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]+sum_of_priorities_interval[3]])
        sum_priorities['high'] = fuzz.trimf(sum_priorities.universe, [sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]+sum_of_priorities_interval[3], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]+sum_of_priorities_interval[3]+sum_of_priorities_interval[4]])
        sum_priorities['very_high'] = fuzz.trapmf(sum_priorities.universe, [sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]+sum_of_priorities_interval[3], sum_of_priorities_interval[0]+sum_of_priorities_interval[1]+sum_of_priorities_interval[2]+sum_of_priorities_interval[3]+sum_of_priorities_interval[4], 3.0, 3.1])

        distance_between_targets['very_close'] = fuzz.trapmf(distance_between_targets.universe, [-1, 0, distance_between_targets_interval[0], distance_between_targets_interval[0]+distance_between_targets_interval[1]])
        distance_between_targets['close'] = fuzz.trimf(distance_between_targets.universe, [distance_between_targets_interval[0], distance_between_targets_interval[0]+distance_between_targets_interval[1], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]])
        distance_between_targets['medium'] = fuzz.trimf(distance_between_targets.universe, [distance_between_targets_interval[0]+distance_between_targets_interval[1], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]+distance_between_targets_interval[3]])
        distance_between_targets['far'] = fuzz.trimf(distance_between_targets.universe, [distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]+distance_between_targets_interval[3], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]+distance_between_targets_interval[3]+distance_between_targets_interval[4]])
        distance_between_targets['very_far'] = fuzz.trapmf(distance_between_targets.universe, [distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]+distance_between_targets_interval[3], distance_between_targets_interval[0]+distance_between_targets_interval[1]+distance_between_targets_interval[2]+distance_between_targets_interval[3]+distance_between_targets_interval[4], 400.0, 400.1])

        pair_priority['very_low'] = fuzz.trapmf(pair_priority.universe, [-1, 0, two_cells_priority_interval[0], two_cells_priority_interval[0]+two_cells_priority_interval[1]])
        pair_priority['low'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0], two_cells_priority_interval[0]+two_cells_priority_interval[1], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]])
        pair_priority['medium'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]+two_cells_priority_interval[3]])
        pair_priority['high'] = fuzz.trimf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]+two_cells_priority_interval[3], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]+two_cells_priority_interval[3]+two_cells_priority_interval[4]])
        pair_priority['very_high'] = fuzz.trapmf(pair_priority.universe, [two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]+two_cells_priority_interval[3], two_cells_priority_interval[0]+two_cells_priority_interval[1]+two_cells_priority_interval[2]+two_cells_priority_interval[3]+two_cells_priority_interval[4], 300.0, 300.1])

        # Sets
        sum_of_priorities_sets = ['very_low', 'low', 'medium', 'high', 'very_high']
        distance_between_targets_sets = ['very_close', 'close', 'medium', 'far', 'very_far']
        two_cell_priority_sets = ['very_low', 'low', 'medium', 'high', 'very_high']  ### 0, 1, .., 4

        ### 5x5 = 25 rules
        idx = 0
        second_system_active_rules = []
        for u in sum_of_priorities_sets:
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
        u_range = np.linspace(0, 2, 50) 
        d_range = np.linspace(0, 30*10, 60)
        
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
        s_range = np.linspace(0, 3.0, 75)
        d_range = np.linspace(0, 30*10, 60)
        
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

    
