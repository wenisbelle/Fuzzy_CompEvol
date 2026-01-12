def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> list:
        
        # 1. Setup Data
        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        rows, cols = map_data.shape

        # ---------------------------------------------------------
        # VECTORIZED DISTANCE CALCULATION
        # ---------------------------------------------------------
        
        # A. Generate a grid of indices [[0,0...], [1,1...]] and [[0,1...], [0,1...]]
        # shape is (2, rows, cols)
        i_grid, j_grid = np.indices((rows, cols))

        # B. Convert grid indices to physical coordinates (Broadcasting)
        # This applies the math to 10,000 cells instantly
        x_grid = i_grid * self.distance_between_cells - map_center_offset
        y_grid = j_grid * self.distance_between_cells - map_center_offset

        # C. Calculate Euclidean Distance Matrix
        # (x_grid - drone_x) subtracts the scalar drone_x from every cell
        dist_matrix = np.sqrt((x_grid - drone_x)**2 + (y_grid - drone_y)**2)

        # ---------------------------------------------------------
        # VECTORIZED UNCERTAINTY
        # ---------------------------------------------------------
        # Clip the entire map at once (removes the "if ind_unc > 2.0" check)
        ind_unc_matrix = np.clip(map_data, 0, 2.0)

        # ---------------------------------------------------------
        # PREPARING INPUTS FOR THE INTERPOLATOR
        # ---------------------------------------------------------
        
        # Flatten the matrices so we have simple lists of numbers
        dist_flat = dist_matrix.ravel()
        unc_flat = ind_unc_matrix.ravel()
        
        # Coordinate tracking (equivalent to your coords_tracker)
        # We stack i and j to get pairs like [(0,0), (0,1), (0,2)...]
        coords_flat = np.column_stack((i_grid.ravel(), j_grid.ravel()))

        # !!! CRITICAL: TRAJECTORY BOTTLENECK !!!
        # The trajectory logic is complex geometry. You likely still need a loop 
        # for this unless you approximate it. 
        # Here we initialize an array for it.
        traj_flat = np.zeros_like(dist_flat)

        # If your trajectory logic is truly "custom" and cannot be vectorized, 
        # you run a simplified loop just to fill this one array:
        for idx in range(len(traj_flat)):
             i, j = coords_flat[idx]
             # ... run your get_cells_visited_in_trajectory(i, j) ...
             # traj_flat[idx] = result
             traj_flat[idx] = 10.0 # Placeholder
        
        # Clip trajectory values (Vectorized)
        traj_flat = np.clip(traj_flat, 0, 30.0)

        # ---------------------------------------------------------
        # FINAL STACKING
        # ---------------------------------------------------------
        # Combine into the (N, 3) shape required by the Interpolator
        # Columns: [trajectory, distance, uncertainty]
        input_array = np.column_stack((traj_flat, dist_flat, unc_flat))

        # Query the Interpolator
        priorities = self.interp_one_cell(input_array)

        # Reformat output to match your original list structure
        # zip() combines the arrays back into tuples
        priority_scores = list(zip(priorities, map(tuple, coords_flat)))

        return priority_scores