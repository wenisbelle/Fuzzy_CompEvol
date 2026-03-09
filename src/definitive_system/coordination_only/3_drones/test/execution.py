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
def create_and_run_simulation(individual):
    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    # Configuring simulation
    config = SimulationConfiguration(
        duration=600, 
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

    #for i in range(MAP_WIDTH):
    #    for j in range(MAP_HEIGHT):
    #        # Assuming that the distance between POI's is 10 meters
    #        x_coord = 10 * i - (MAP_WIDTH * 10) / 2
    #        y_coord = 10 * j - (MAP_HEIGHT * 10) / 2
    #        builder.add_node(PointOfInterest,
    #                         (x_coord, y_coord, 0))    

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
        filename=f'logs/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    for _ in range(1):
        individual =  [np.float64(0.1440649407269286), np.float64(0.21641969548096338), np.float64(0.29890579865235795),
                       np.float64(26.105537532686153), np.float64(38.18495818334767), np.float64(34.01857399947521),
                       np.float64(0.3002687008423602), np.float64(0.3107887710774788), np.float64(0.15979961222311245),
                       np.float64(0.17241184562715162), np.float64(0.5078836540840643), np.float64(0.6585830974598943),
                       np.float64(21.8317388375936), np.float64(42.64213912794975), np.float64(18.05825870767923),
                       np.float64(0.2927583897307974), np.float64(0.37619404958239544), np.float64(0.23413715772407398),
                       np.int64(0), np.int64(0), np.int64(0),
                       np.int64(2), np.int64(1), np.int64(0),
                       np.int64(4), np.int64(3), 3,
                       np.int64(0),0, np.int64(1),
                       np.int64(1),np.int64(2), np.int64(3),
                       np.int64(1),np.int64(3), np.int64(4)]
        create_and_run_simulation(individual)
    


if __name__ == "__main__":
    main()