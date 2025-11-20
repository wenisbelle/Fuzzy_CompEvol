import numpy as np
from typing import Tuple, List

def get_cells_visited_in_trajectory(
    uav_altitude: float, 
    initial_cell: Tuple[int, int], 
    final_cell: Tuple[int, int],
    cell_size_meters: float = 10.0  # Pass cell size as a parameter
) -> list:
    
    cells_within_trajectory = []
    # This radius is in "meters" (or whatever unit altitude is)
    radius_coverage = uav_altitude * np.tan(np.pi/6) # approx 10 * 0.577 = 5.77m
    print(f"Coverage Radius: {radius_coverage:.2f} meters")

    if final_cell[0] == initial_cell[0]:  # Vertical line case
        print("Vertical line case")
        for y in range(min(initial_cell[1], final_cell[1])+1, max(initial_cell[1], final_cell[1])+1):
            cells_within_trajectory.append((initial_cell[0], y))
        return cells_within_trajectory
    
    elif final_cell[1] == initial_cell[1]:  # Horizontal line case
        print("Horizontal line case")
        for x in range(min(initial_cell[0], final_cell[0])+1, max(initial_cell[0], final_cell[0])+1):
            cells_within_trajectory.append((x, initial_cell[1]))
        return cells_within_trajectory

    else:
        print("General case")
        line_slop = (final_cell[1] - initial_cell[1]) / (final_cell[0] - initial_cell[0] + 1e-6)
        print(f"Line slop: {line_slop:.2f}")

        # Note: These loops check every cell in the bounding box
        # This is inefficient but correct for finding a "corridor"
        for y in range(min(initial_cell[1], final_cell[1]), max(initial_cell[1], final_cell[1]) + 1):
            for x in range(min(initial_cell[0], final_cell[0]), max(initial_cell[0], final_cell[0]) + 1):
                
                # Skip the initial cell itself
                if x == initial_cell[0] and y == initial_cell[1]:
                    continue

                ## distance from point to line formula ##
                # d is in "cell units"
                d_cells = abs(line_slop*x - y + (initial_cell[1] - line_slop*initial_cell[0])) / np.sqrt(line_slop**2 + 1)
                
                # Convert distance from "cell units" to "meters"
                d_meters = d_cells * cell_size_meters
                
                # Compare meters to meters
                if d_meters <= radius_coverage:
                    cells_within_trajectory.append((x, y))
                    
        return cells_within_trajectory 


def main():
    map_shape = (10, 10)
    map_to_visualize = np.zeros(map_shape)
    
    drone_position = [0, 0, 10] # x=0, y=0, z=10
    destination = [8, 9]       # x=3, y=9
    cell_size = 10.0           # Each cell is 10x10 meters

    initial_cell_coords = (drone_position[0], drone_position[1])
    final_cell_coords = (destination[0], destination[1])

    cells_covered = get_cells_visited_in_trajectory(
            uav_altitude=drone_position[2],
            initial_cell=initial_cell_coords,
            final_cell=final_cell_coords,
            cell_size_meters=cell_size
        )
    
    print(f"\nCells covered in trajectory: {cells_covered}")

    # --- VISUALIZATION ---
    # Mark covered cells with a 1
    for cell_xy in cells_covered:
        x, y = cell_xy
        if 0 <= x < map_shape[1] and 0 <= y < map_shape[0]: # Check bounds
            map_to_visualize[y, x] = 1  # Note: numpy is (row, col) -> (y, x)

    # Mark start (8) and end (9) points
    map_to_visualize[initial_cell_coords[1], initial_cell_coords[0]] = 8
    map_to_visualize[final_cell_coords[1], final_cell_coords[0]] = 9

    print("\nVisualized Trajectory (8=Start, 9=End, 1=Covered):")
    print(map_to_visualize)


if __name__ == "__main__":
    main()