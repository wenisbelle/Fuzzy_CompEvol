import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .test_inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np

#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual):
    # Configuring simulation
    config = SimulationConfiguration(
        duration=250, 
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
        uncertainty_rate=0.05,
        vanishing_update_time=10.0,
        trajectory_accomulate_fitness_norm=individual[0],
        distance_norm=individual[1],
        uncertainty_norm=individual[2],
        distance_between_drone_norm=individual[3],
        number_of_drones=3,
        map_width=10,
        map_height=10,
        results_aggregator=results_aggregator
    )

    for _ in range(3):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

    map_width = 10
    map_height = 10
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

    total_uncertainty_drone1 = results_aggregator[0]['accomulated_uncertainty']
    total_uncertainty_drone2 = results_aggregator[1]['accomulated_uncertainty']
    total_uncertainty_drone3 = results_aggregator[2]['accomulated_uncertainty']


    return (total_uncertainty_drone1+total_uncertainty_drone2+total_uncertainty_drone3)/3

#### Main to test the best individual found #####
def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'surveillance_v3/logs/test_ga/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )

    best_individual = [0.85, 692.02, 64.97, 39.58]
    
    for _ in range(10):
        final_uncertainty = create_and_run_simulation(best_individual)
        print(f"Final uncertainty with best individual {best_individual}: {final_uncertainty}")

if __name__ == "__main__":
    main()
