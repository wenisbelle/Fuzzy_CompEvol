import numpy as np
from typing import Tuple, List
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import RegularGridInterpolator

class FuzzyEvaluatorFast:
    def __init__(self, map_width: int, map_height: int, camera_angle: float, distance_between_cells: int = 10):
        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells

        # 1. Setup the Fuzzy Systems (Same as your code)
        self._setup_fuzzy_systems()

        # 2. GENERATE LOOKUP TABLES (The Optimization)
        print("Pre-computing fuzzy control surfaces... (this happens once)")
        self.interp_one_cell = self._precompute_one_cell_surface()
        self.interp_two_cells = self._precompute_two_cells_surface()
        print("Pre-computation complete.")

    def _setup_fuzzy_systems(self):
        """Initializes the skfuzzy control systems (Internal use only)"""
        # --- System 1: One Cell ---
        uncertainty = ctrl.Antecedent(np.arange(0, 30, 0.5), 'uncertainty')
        distance = ctrl.Antecedent(np.arange(0, 150, 5.0), 'distance')
        ind_uncertainty = ctrl.Antecedent(np.arange(0, 2.0, 0.05), 'individual_cell_uncertainty')
        one_cell_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'one_cell_priority', defuzzify_method='centroid')

        # Membership Functions (Copied from your code)
        uncertainty['low'] = fuzz.trapmf(uncertainty.universe, [-1, 0, 2, 5])
        uncertainty['medium'] = fuzz.trimf(uncertainty.universe, [2, 5, 7])
        uncertainty['high'] = fuzz.trapmf(uncertainty.universe, [5, 7, 30, 31])
        
        distance['close'] = fuzz.trapmf(distance.universe, [-1, 0, 40, 80])
        distance['medium'] = fuzz.trimf(distance.universe, [40, 80, 120])
        distance['far'] = fuzz.trapmf(distance.universe, [80, 120, 150, 151])

        ind_uncertainty['low'] = fuzz.trapmf(ind_uncertainty.universe, [-0.1, 0.0, 0.25, 0.5])
        ind_uncertainty['medium'] = fuzz.trimf(ind_uncertainty.universe, [0.25, 0.5, 0.75])
        ind_uncertainty['high'] = fuzz.trapmf(ind_uncertainty.universe, [0.5, 0.75, 2.0, 2.1])

        one_cell_priority['very_low'] = fuzz.trimf(one_cell_priority.universe, [-0.1, 0.0, 0.25])
        one_cell_priority['low'] = fuzz.trimf(one_cell_priority.universe, [0.0, 0.25, 0.5])
        one_cell_priority['medium'] = fuzz.trimf(one_cell_priority.universe, [0.25, 0.5, 0.75])
        one_cell_priority['high'] = fuzz.trimf(one_cell_priority.universe, [0.5, 0.75, 1.0])
        one_cell_priority['very_high'] = fuzz.trimf(one_cell_priority.universe, [0.75, 1.0, 1.1])

        # Rules System 1
        rules1 = [
            ctrl.Rule(uncertainty['high'] & distance['close'], one_cell_priority['very_high']),
            ctrl.Rule(uncertainty['high'] & distance['medium'], one_cell_priority['very_high']),
            ctrl.Rule(uncertainty['high'] & distance['far'], one_cell_priority['medium']),
            ctrl.Rule(uncertainty['medium'] & distance['close'] & ind_uncertainty['high'], one_cell_priority['very_high']),
            ctrl.Rule(uncertainty['medium'] & distance['close'] & ind_uncertainty['medium'], one_cell_priority['medium']),
            ctrl.Rule(uncertainty['medium'] & distance['close'] & ind_uncertainty['low'], one_cell_priority['very_low']),
            ctrl.Rule(uncertainty['medium'] & distance['medium'] & ind_uncertainty['high'], one_cell_priority['high']),
            ctrl.Rule(uncertainty['medium'] & distance['medium'] & ind_uncertainty['medium'], one_cell_priority['medium']),
            ctrl.Rule(uncertainty['medium'] & distance['medium'] & ind_uncertainty['low'], one_cell_priority['very_low']),
            ctrl.Rule(uncertainty['medium'] & distance['far'] & ind_uncertainty['high'], one_cell_priority['medium']),
            ctrl.Rule(uncertainty['medium'] & distance['far'] & ind_uncertainty['medium'], one_cell_priority['low']),
            ctrl.Rule(uncertainty['medium'] & distance['far'] & ind_uncertainty['low'], one_cell_priority['very_low']),
            ctrl.Rule(uncertainty['low'] & distance['close'] & ind_uncertainty['high'], one_cell_priority['medium']),
            ctrl.Rule(uncertainty['low'] & distance['close'] & ind_uncertainty['medium'], one_cell_priority['low']),
            ctrl.Rule(uncertainty['low'] & distance['close'] & ind_uncertainty['low'], one_cell_priority['very_low']),
            ctrl.Rule(uncertainty['low'] & distance['medium'] & ind_uncertainty['high'], one_cell_priority['low']),
            ctrl.Rule(uncertainty['low'] & distance['medium'] & ind_uncertainty['medium'], one_cell_priority['very_low']),
            ctrl.Rule(uncertainty['low'] & distance['medium'] & ind_uncertainty['low'], one_cell_priority['very_low']),
            ctrl.Rule(uncertainty['low'] & distance['far'], one_cell_priority['very_low'])
        ]
        self.one_cell_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules1))

        # --- System 2: Two Cells ---
        sum_priorities = ctrl.Antecedent(np.arange(0, 2.0, 0.01), 'sum_priorities')
        dist_targets = ctrl.Antecedent(np.arange(0, 150, 1.0), 'distance_between_targets')
        pair_priority = ctrl.Consequent(np.arange(0, 1.0, 0.01), 'pair_priority', defuzzify_method='centroid')

        sum_priorities['low'] = fuzz.trapmf(sum_priorities.universe, [-0.1, 0.0, 0.5, 1.0])
        sum_priorities['medium'] = fuzz.trimf(sum_priorities.universe, [0.5, 1.0, 1.5])
        sum_priorities['high'] = fuzz.trapmf(sum_priorities.universe, [1.0, 1.5, 2.0, 2.1])

        dist_targets['close'] = fuzz.trapmf(dist_targets.universe, [-1, 0, 40, 80])
        dist_targets['medium'] = fuzz.trimf(dist_targets.universe, [40, 80, 120])
        dist_targets['far'] = fuzz.trapmf(dist_targets.universe, [80, 120, 150, 151])

        pair_priority['very_low'] = fuzz.trimf(pair_priority.universe, [-0.1, 0.0, 0.25])
        pair_priority['low'] = fuzz.trimf(pair_priority.universe, [0.0, 0.25, 0.50])
        pair_priority['medium'] = fuzz.trimf(pair_priority.universe, [0.25, 0.50, 0.75])
        pair_priority['high'] = fuzz.trimf(pair_priority.universe, [0.50, 0.75, 1.0])
        pair_priority['very_high'] = fuzz.trimf(pair_priority.universe, [0.75, 1.0, 1.1])

        rules2 = [
            ctrl.Rule(sum_priorities['high'] & dist_targets['far'], pair_priority['very_high']),
            ctrl.Rule(sum_priorities['high'] & dist_targets['medium'], pair_priority['very_high']),
            ctrl.Rule(sum_priorities['high'] & dist_targets['close'], pair_priority['very_low']),
            ctrl.Rule(sum_priorities['medium'] & dist_targets['far'], pair_priority['high']),
            ctrl.Rule(sum_priorities['medium'] & dist_targets['medium'], pair_priority['medium']),
            ctrl.Rule(sum_priorities['medium'] & dist_targets['close'], pair_priority['very_low']),
            ctrl.Rule(sum_priorities['low'] & dist_targets['far'], pair_priority['low']),
            ctrl.Rule(sum_priorities['low'] & dist_targets['medium'], pair_priority['very_low']),
            ctrl.Rule(sum_priorities['low'] & dist_targets['close'], pair_priority['very_low'])
        ]
        self.two_cells_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules2))

    def _precompute_one_cell_surface(self):
        """Generates a 3D Lookup Table for the first fuzzy system."""
        # Define the resolution of your lookup table (fewer points = faster init, more points = more accuracy)
        # We use a bit coarser steps than the universe definition to save memory, interpolate handles the rest
        u_range = np.linspace(0, 30, 30) 
        d_range = np.linspace(0, 150, 30)
        i_range = np.linspace(0, 2.0, 20)
        
        # Create a grid
        output_surface = np.zeros((len(u_range), len(d_range), len(i_range)))

        # Fill the grid (This part is slow, but only runs ONCE at startup)
        for i, u_val in enumerate(u_range):
            for j, d_val in enumerate(d_range):
                for k, i_val in enumerate(i_range):
                    self.one_cell_sim.input['uncertainty'] = u_val
                    self.one_cell_sim.input['distance'] = d_val
                    self.one_cell_sim.input['individual_cell_uncertainty'] = i_val
                    self.one_cell_sim.compute()
                    output_surface[i, j, k] = self.one_cell_sim.output['one_cell_priority']
        
        # Create the interpolator function
        return RegularGridInterpolator((u_range, d_range, i_range), output_surface, bounds_error=False, fill_value=None)

    def _precompute_two_cells_surface(self):
        """Generates a 2D Lookup Table for the second fuzzy system."""
        s_range = np.linspace(0, 2.0, 40)
        d_range = np.linspace(0, 150, 40)
        
        output_surface = np.zeros((len(s_range), len(d_range)))

        for i, s_val in enumerate(s_range):
            for j, d_val in enumerate(d_range):
                self.two_cells_sim.input['sum_priorities'] = s_val
                self.two_cells_sim.input['distance_between_targets'] = d_val
                self.two_cells_sim.compute()
                output_surface[i, j] = self.two_cells_sim.output['pair_priority']

        return RegularGridInterpolator((s_range, d_range), output_surface, bounds_error=False, fill_value=None)

    # -------------------------------------------------------------
    # OPTIMIZED RUNTIME METHODS
    # -------------------------------------------------------------

    def get_cells_visited_in_trajectory(self, drone_altitude: float, initial_cell: Tuple[int, int], final_cell: Tuple[int, int]) -> list:
        # (This helper remains mostly the same, it involves geometric logic 
        # that is hard to vectorize without rasterization libraries)
        cells_within_trajectory = []
        radius_coverage = drone_altitude * np.tan(self.camera_angle)
        
        # ... [Keep your existing logic for line tracing here] ...
        # For brevity, assuming this method exists as you wrote it.
        # Ideally, move the implementation here.
        return [] # Placeholder for the original logic

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

                # Trajectory (This logic remains your specific custom logic)
                # You didn't provide the full logic for 'get_cells_visited', 
                # so assuming you run it here to get 'trajectory_value'
                # ... [Run get_cells_visited_in_trajectory here] ...
                trajectory_value = 10.0 # Placeholder: Insert your trajectory logic result here
                
                # Individual Uncertainty
                ind_unc = map_data[i, j]
                if ind_unc > 2.0: ind_unc = 2.0
                if trajectory_value > 30: trajectory_value = 30

                input_data.append([trajectory_value, dist, ind_unc])
                coords_tracker.append((i, j))

        # 2. Vectorized Fuzzy Inference (The Speedup)
        # Convert to numpy array
        input_array = np.array(input_data)
        
        # Query the Lookup Table (Instant)
        # interpolator expects shape (N, 3)
        priorities = self.interp_one_cell(input_array)

        # 3. Format Output
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
        # Data structure: [priority, (x, y)]
        # We need to separate priority values and coordinates
        p1_vals = np.array([x[0] for x in list1])
        p1_coords = np.array([x[1] for x in list1]) # Shape (N, 2)

        p2_vals = np.array([x[0] for x in list2])
        p2_coords = np.array([x[1] for x in list2]) # Shape (M, 2)

        # 1. Vectorized Sum of Priorities
        # Shape (N, 1) + (1, M) -> (N, M)
        sum_p_matrix = p1_vals[:, np.newaxis] + p2_vals[np.newaxis, :]

        # 2. Vectorized Distance Calculation
        # Convert grid indices to physical coordinates
        phys_p1 = (p1_coords * self.distance_between_cells) - map_center_offset
        phys_p2 = (p2_coords * self.distance_between_cells) - map_center_offset

        # Broadcasting distance: (N, 1, 2) - (1, M, 2)
        diff = phys_p1[:, np.newaxis, :] - phys_p2[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2)) # Shape (N, M)

        # 3. Prepare inputs for Interpolator
        # We need to flatten the matrices to feed the interpolator a list of points
        # Input shape needs to be (N*M, 2) -> columns: [sum_priority, distance]
        
        flat_sum = sum_p_matrix.ravel()
        flat_dist = dist_matrix.ravel()
        
        input_stack = np.column_stack((flat_sum, flat_dist))

        # 4. Query the Lookup Table (Instant)
        combined_priorities = self.interp_two_cells(input_stack)

        # 5. Reconstruct the list structure
        # We need indices to know which cells generated which priority
        n_indices, m_indices = np.indices(sum_p_matrix.shape)
        flat_n = n_indices.ravel()
        flat_m = m_indices.ravel()

        combined_priority_scores = []
        # This loop is just for repackaging, the heavy math is done
        for k in range(len(combined_priorities)):
             # Access original lists to get cell coordinates back
             cell1 = list1[flat_n[k]][1]
             cell2 = list2[flat_m[k]][1]
             score = combined_priorities[k]
             combined_priority_scores.append((score, cell1, cell2))

        return combined_priority_scores