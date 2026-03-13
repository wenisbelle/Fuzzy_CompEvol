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
    NUMBER_OF_DRONES = 5

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
        filename=f'definitive_system/coordination_only/5_drones/test/logs/fuzzy_2000_modified/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual = [np.float64(0.09597064286042159), np.float64(0.2563172828950639), np.float64(0.3959936959674909),
                  np.float64(104.37352913819454), np.float64(132.8924071003155), np.float64(52.57694971352459),
                  np.float64(0.237393697366025), np.float64(0.23569208943791883), np.float64(0.23650170258746464),
                  np.float64(0.4242592879838075), np.float64(0.11757950858765165), np.float64(0.7279060144523078),
                  np.float64(48.001097515887565), np.float64(41.77598292899479), np.float64(42.71703141372458),
                  np.float64(0.03494489683012761), np.float64(0.1610053383887725), np.float64(0.37883516875280976),
                  np.int64(0), np.int64(0), np.int64(0),
                  np.int64(3), np.int64(2), np.int64(0),
                  np.int64(4), np.int64(3), np.int64(2),
                  np.int64(0), np.int64(0), 1,
                  np.int64(1), np.int64(2), 3,
                  2, np.int64(3), np.int64(4)]


    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()

    for _ in range(20):
        create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)    


if __name__ == "__main__":
    main()