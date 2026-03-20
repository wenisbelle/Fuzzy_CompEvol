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
    individual = [np.float64(0.0981836852626658), np.float64(0.39823123425496115), np.float64(0.16054817954314526),
            np.float64(83.89382541378083), np.float64(83.48405608641099), np.float64(80.19027847452021),
            np.float64(0.08537202583746739), np.float64(0.22064110276808918), np.float64(0.08468486491055341),
            np.float64(0.781600537288768), np.float64(0.14279249103885555), np.float64(0.6683300577689935),
            np.float64(50.383763478430325), np.float64(40.049383832055256), np.float64(45.49243243462786),
            np.float64(0.2736919999205339), np.float64(0.34228046734274264), np.float64(0.1823828011916267),
            np.int64(2), np.int64(2), np.int64(1),
            np.int64(3), np.int64(3), 2,
            np.int64(4),np.int64(4), np.int64(3),
            np.int64(3),np.int64(1), np.int64(3),
            np.int64(2),np.int64(1), np.int64(3),
            np.int64(0), np.int64(3), np.int64(4)]




    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()

    for _ in range(10):
        create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)    


if __name__ == "__main__":
    main()