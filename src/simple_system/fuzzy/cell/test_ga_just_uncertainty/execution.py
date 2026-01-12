import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .fuzzy_inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np


#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual):
    # Configuring simulation
    config = SimulationConfiguration(
        duration=150, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    #builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))


    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=0.05,
        vanishing_update_time=10.0,
        number_of_drones=3,
        map_width=10,
        map_height=10,
        fuzzy_parameters=np.array(individual),
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
    medium_uncertainty = (total_uncertainty_drone1+total_uncertainty_drone2+total_uncertainty_drone3)/3
    print(f"Variable to be minimized: {medium_uncertainty}")
    return medium_uncertainty


def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'new_system/logs/fuzzy/test_ga_just_uncertainty/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual = [np.float64(0.2452756117192072), np.float64(0.6314438801204216), np.float64(0.14896525889966167), np.float64(41.93558185495538), np.float64(48.555596666024314), np.float64(40.089918301854205), np.float64(0.1331198108939456), np.float64(0.1813704389233992), np.float64(0.40002804167464495), np.float64(0.8466360088718689), np.float64(0.0747009832397093), np.float64(0.1721806812773743), np.float64(35.50476953613601), np.float64(51.0127129434542), np.float64(55.34351234187098), np.float64(0.06657950052596032), np.float64(0.18701170141853954), np.float64(0.3796480632136605)]

    for _ in range(10):
        create_and_run_simulation(individual)
    


if __name__ == "__main__":
    main()
