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
        duration=1000, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler())
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

    total_uncertainty_drone1 = results_aggregator[0]['accomulated_uncertainty']
    total_uncertainty_drone2 = results_aggregator[1]['accomulated_uncertainty']
    total_uncertainty_drone3 = results_aggregator[2]['accomulated_uncertainty']
    medium_uncertainty = 0.01*(total_uncertainty_drone1+total_uncertainty_drone2+total_uncertainty_drone3)/3
    print(f"Variable to be minimized: {medium_uncertainty}")
    
    return medium_uncertainty


def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'logs/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual =  [np.float64(0.009598218750528936), np.float64(0.38309031666688365), np.float64(0.5464715792450225), np.float64(59.46087613645159), np.float64(21.901328499990107), np.float64(51.486265776138836), np.float64(0.21425141782193335), np.float64(0.14579579001118176), np.float64(0.2543935518834503), np.float64(0.41202899669965515), np.float64(0.6502140585224361), np.float64(0.33818087613615844), np.float64(11.785379640652145), np.float64(39.42669012151829), np.float64(29.617132344080574), np.float64(0.2684364094637851), np.float64(0.20961702211472472), np.float64(0.20529641794914164), np.int64(1), np.int64(2), np.int64(2), np.int64(0), np.int64(2), np.int64(2), np.int64(3), np.int64(4), 3, np.int64(0), 3, 0, np.int64(0), np.int64(1), np.int64(3), np.int64(3), 3, np.int64(4)]

    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    for _ in range(20):
        create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)
    


if __name__ == "__main__":
    main()