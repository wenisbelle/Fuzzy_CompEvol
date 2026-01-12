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
        filename=f'new_system/logs/fuzzy/test_ga_uncertainty_distance2/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual = [np.float64(0.21993720934244318), np.float64(0.41463881725319884), np.float64(0.33713808032073245), np.float64(31.434394755910724), np.float64(32.850273464509065), np.float64(32.96066484214472), np.float64(0.20255617118579325), np.float64(0.3348156255849202), np.float64(0.26840621054870106), np.float64(0.6235120747607585), np.float64(0.548921650741627), np.float64(0.5034265023541958), np.float64(39.19750152518405), np.float64(33.00739093438502), np.float64(21.46814332166222), np.float64(0.16645243125371706), np.float64(0.27495723128939953), np.float64(0.2998979740995638)]
    
    for _ in range(10):
        create_and_run_simulation(individual)
    


if __name__ == "__main__":
    main()
