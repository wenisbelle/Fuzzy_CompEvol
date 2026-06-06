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
    NUMBER_OF_DRONES = 10

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

    medium_uncertainty = 0
    for i in range(NUMBER_OF_DRONES):
        medium_uncertainty += 0.01*results_aggregator[i]['accomulated_uncertainty']/NUMBER_OF_DRONES

    print(f"Variable to be minimized: {medium_uncertainty}")    
    return medium_uncertainty


def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'definitive_system/coordination_only/10_drones/test/logs/fuzzy_2000/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual =  [np.float64(0.08879789658449988), np.float64(0.2990584261575781), np.float64(0.32384850500067697),
            np.float64(59.67149923152216), np.float64(74.61556171321105), np.float64(87.84193135295521),
            np.float64(0.2771856390455823), np.float64(0.2787134317315393), np.float64(0.1638125494958381),
            np.float64(0.673782021688156), np.float64(0.6329466169294924), np.float64(0.5591202075074384),
            np.float64(50.54463178241406), np.float64(42.27566047833311), np.float64(44.5656822753396),
            np.float64(0.32012394689951496), np.float64(0.2040906672593554), np.float64(0.280814060613372),
            np.int64(0), np.int64(2), np.int64(1),
            np.int64(0), np.int64(4), np.int64(1),
            np.int64(3), np.int64(4), np.int64(3),
            np.int64(1), np.int64(0), np.int64(2),
            np.int64(2), np.int64(3), np.int64(3),
            np.int64(4), np.int64(3), np.int64(4)] 

    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    for _ in range(20):
        create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)
    
if __name__ == "__main__":
    main()