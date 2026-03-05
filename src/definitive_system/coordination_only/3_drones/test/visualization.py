import matplotlib.pyplot as plt
import numpy as np
import logging

class MapVisualizer:
    """
    Handles the real-time visualization of drone maps using Matplotlib.
    """
    def __init__(self, num_drones: int, map_size: int = 100, threshold: float = 0.5):
        try:
            plt.ion()
            
            # FIXED: squeeze=False forces self.axes to always be a 2D array (1, num_drones)
            self.fig, self.axes = plt.subplots(1, num_drones, figsize=(5 * num_drones, 10), squeeze=False)

            self.map_shape = (int(np.sqrt(map_size)), int(np.sqrt(map_size)))
            self.images = [] 
            
            for i in range(num_drones):
                initial_map_data = np.ones(self.map_shape)

                # This is now perfectly safe because of squeeze=False
                ax_top = self.axes[0, i]
                im_top = ax_top.imshow(initial_map_data, cmap='gray_r', vmin=0, vmax=1, origin='lower')
                ax_top.set_title(f"Drone {i} Map")
                ax_top.set_xticks([])
                ax_top.set_yticks([])
                self.images.append(im_top)
                
            self.fig.tight_layout(pad=2.0)
            plt.show()
            
        except Exception as e:
            # If this triggers, your drone maps will not work!
            logging.error(f"Error initializing visualizer: {e}")
            raise # It's usually better to raise the error so you know it failed

    def update_map(self, drone_id: int, map_data: np.ndarray):
        """
        Updates the map visualization for a specific drone.
        """
        try:
            plot_index = drone_id
            
            # 1. Extract ONLY the first column (your actual map pixel data)
            pixel_data = map_data[:, 0]
            
            # 2. Safely clip the values between 0 and 1 (replaces your old for-loop)
            pixel_data = np.clip(pixel_data, 0, 1)
            
            # 3. Reshape the 1D array back into the 2D grid (e.g., 100x100)
            map_view = pixel_data.reshape(self.map_shape)
            
            # 4. Update the plot with the correctly shaped 2D array
            self.images[plot_index].set_data(map_view)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logging.warning(f"Could not update map for drone {drone_id}: {e}")
        
    def close(self):
        """Closes the Matplotlib window."""
        plt.ioff()
        plt.close(self.fig)