import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium


def create_and_run_simulation(drone_params: dict):
    # Configuring simulation
    config = SimulationConfiguration(
        duration=50, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))


    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=drone_params["uncertainty_rate"],
        vanishing_update_time=drone_params["vanishing_update_time"],
        map_threshold=drone_params["map_threshold"],
        distance_norm=drone_params["distance_norm"],
        cluster_size_norm=drone_params["cluster_size_norm"],
        number_of_drones=drone_params["number_of_drones"],
        map_width=drone_params["map_width"],
        map_height=drone_params["map_height"],
        results_aggregator=results_aggregator
    )

    for _ in range(drone_params["number_of_drones"]):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

    map_width = drone_params["map_width"]
    map_height = drone_params["map_height"]
    for i in range(map_width):
        for j in range(map_height):
            # Assuming the coordinate logic is (10*i-50, 10*j-50, 0)
            # based on the original 10x10 map
            x_coord = 10 * i - (map_width * 10) / 2
            y_coord = 10 * j - (map_height * 10) / 2
            builder.add_node(PointOfInterest,
                             (x_coord, y_coord, 0))    

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()
    
    return results_aggregator


def main():
    # Configure logging to write to a file
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'surveillance_v3/logs/inteligent_protocol_all_cells/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    drone_params = {
        "uncertainty_rate": 0.05,
        "vanishing_update_time": 10.0,
        "map_threshold": 0.5,
        "distance_norm": 200,
        "cluster_size_norm": 1,
        "number_of_drones": 2,    
        "map_width": 10,
        "map_height": 10
    }

    NUMBER_OF_RUNS = 1
    for i in range(NUMBER_OF_RUNS):
        print(f"Starting run {i+1}/{NUMBER_OF_RUNS}...")
        simulation_results = create_and_run_simulation(drone_params)        
        logging.info(f"Run {i+1} results: {simulation_results}")
        print(f"Run {i+1} finished. Results: {simulation_results}")

if __name__ == "__main__":
    main()
