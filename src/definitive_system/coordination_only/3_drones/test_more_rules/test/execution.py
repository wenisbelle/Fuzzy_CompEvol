import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .lookup_table_generator import FuzzyLookupTable
from .fuzzy_inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np


#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(lookup_one_cell, lookup_two_cells):
        
    # Configuring simulation
    config = SimulationConfiguration(
        duration=2000, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    #builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=200
    )))

    MAP_WIDTH = 50
    MAP_HEIGHT = 50
    NUMBER_OF_DRONES = 3

    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=0.01,
        vanishing_update_time=10.0,
        number_of_drones=NUMBER_OF_DRONES,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        fuzzy_tables=[lookup_one_cell, lookup_two_cells],
        results_aggregator=results_aggregator
    )

    for _ in range(NUMBER_OF_DRONES):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

  

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

    total_uncertainty_drone1 = results_aggregator[0]['accomulated_uncertainty']
    total_uncertainty_drone2 = results_aggregator[1]['accomulated_uncertainty']
    total_uncertainty_drone3 = results_aggregator[2]['accomulated_uncertainty']
    medium_uncertainty = 0.01*(total_uncertainty_drone1+total_uncertainty_drone2+total_uncertainty_drone3)/3

    print(f"Variable to be minimized: {medium_uncertainty}")
    
    return medium_uncertainty


def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'definitive_system/coordination_only/3_drones/test_more_rules/test/logs/fuzzy_2000/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual =  [np.float64(0.14096521481670332), np.float64(0.6302666709175262), np.float64(0.23828589684651877), np.float64(76.9642128330259), np.float64(73.09834769417856), np.float64(86.48930169835975), np.float64(0.27770662335759483), np.float64(0.27961929474353586), np.float64(0.2506011356399168), np.float64(0.5591706633738546), np.float64(0.7888351550206498), np.float64(0.48787776176527164), np.float64(60.30931111169186), np.float64(35.97497838548361), np.float64(50.676817397372915), np.float64(0.10247093805588445), np.float64(0.16016706177221007), np.float64(0.18517072054655448), np.int64(2), np.int64(2), np.int64(0), 4, np.int64(3), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(2), np.int64(2), 0, np.int64(2), 4, np.int64(4), np.int64(3), np.int64(0), np.int64(0)]
    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    for _ in range(10):
                create_and_run_simulation(lookup_one_cell, lookup_two_cells)    

if __name__ == "__main__":
    main()