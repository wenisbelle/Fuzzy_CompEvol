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
        filename=f'definitive_system/coordination_only/3_drones/test/logs/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    individual = [np.float64(0.3016017252572088), np.float64(0.3689492662528065), np.float64(0.16615127867290264), np.float64(64.87864115655191), np.float64(76.145066071908), np.float64(91.31303192923832), np.float64(0.18813807055184711), np.float64(0.18079248076861443), np.float64(0.10863734568525232), np.float64(0.505080320416538), np.float64(0.3371722189839078), np.float64(0.3276336576734477), np.float64(39.7996092540598), np.float64(28.364776785809596), np.float64(41.932421470701456), np.float64(0.2868023793768918), np.float64(0.13232743595774496), np.float64(0.06696275294570064), np.int64(1), 2, np.int64(2), np.int64(1), 0, np.int64(3), 4, 4, 0, 2, np.int64(0), np.int64(4), np.int64(3), np.int64(2), np.int64(4), np.int64(4), np.int64(2), np.int64(1)]

    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    for _ in range(20):
                create_and_run_simulation(individual, lookup_one_cell, lookup_two_cells)    

if __name__ == "__main__":
    main()