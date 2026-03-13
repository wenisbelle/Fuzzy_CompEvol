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
def create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells):
        
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
        filename=f'definitive_system/coordination_only/3_drones/test/logs/fuzzy_modified_2000/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual = [np.float64(0.1765550275716401), np.float64(0.26504925419859754), np.float64(0.5328280594467583),
                  np.float64(65.76913130640668), np.float64(60.579250249961284), np.float64(86.06840306949528),
                  np.float64(0.09459789068662158), np.float64(0.2810259583239695), np.float64(0.31808056721276934),
                  np.float64(0.2109100989454195), np.float64(0.5282720591027129), np.float64(0.8521749796977374),
                  np.float64(34.58815897497721), np.float64(27.237106487346836), np.float64(56.10639222453888),
                  np.float64(0.15270903952109194), np.float64(0.21795421097999682), np.float64(0.23102459648581986),
                  np.int64(1), np.int64(1), np.int64(0),
                  np.int64(2), 1, np.int64(2),
                  np.int64(4), np.int64(3), np.int64(1),
                  np.int64(0), np.int64(1), np.int64(1),
                  np.int64(2), np.int64(2), np.int64(3),
                  np.int64(2), np.int64(3), np.int64(4)]
    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    for _ in range(5):
                create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)    

if __name__ == "__main__":
    main()