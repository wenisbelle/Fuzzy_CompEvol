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
    individual =  [np.float64(0.13178514521050166), np.float64(0.22052666462927448), np.float64(0.1833223146881966),
                   np.float64(93.09861893576988), np.float64(109.73193263405224), np.float64(62.63335295427316),
                   np.float64(0.24051449596701177), np.float64(0.22396387718086452), np.float64(0.28687473624994975),
                   np.float64(0.9308023066008706), np.float64(0.07933787608140352), np.float64(0.35790689192113645),
                   np.float64(46.57233582664971), np.float64(34.09404197285841), np.float64(46.356999913792095),
                   np.float64(0.23034430043956328), np.float64(0.34214931547671845), np.float64(0.30865084755944355),
                   np.int64(1), np.int64(1), np.int64(0),
                   np.int64(3), 3, 0,
                   np.int64(4), np.int64(3), 0,
                   np.int64(1), np.int64(0), np.int64(1),
                   np.int64(0), np.int64(4), 3,
                   2, np.int64(4), 4]

    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    for _ in range(20):
        create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)
    
if __name__ == "__main__":
    main()