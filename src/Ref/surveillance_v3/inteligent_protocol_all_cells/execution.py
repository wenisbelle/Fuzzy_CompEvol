import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np

#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual):
    # Configuring simulation
    config = SimulationConfiguration(
        duration=500, 
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
        trajectory_accomulate_fitness_norm=individual[0],
        distance_norm=individual[1],
        uncertainty_norm=individual[2],
        distance_between_drone_norm=individual[3],
        number_of_drones=3,
        map_width=10,
        map_height=10,
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


    return (total_uncertainty_drone1+total_uncertainty_drone2+total_uncertainty_drone3)/3

########### GA part ##########
def objective_function(individual):
    return create_and_run_simulation(individual),


def random_distance_norm():
    return random.uniform(0.0,1000.0)

def random_norm_values():
    return random.uniform(0.0, 200.0)


def main():
    ### Defining the GA ###
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) ## Minimize the accomulated uncertainty 
    creator.create("Individual", list,  fitness=creator.FitnessMin) ## individual
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initCycle, creator.Individual, [random_norm_values, random_distance_norm, random_norm_values, random_norm_values], 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=20)                            
    hof = tools.HallOfFame(1)                                
    stats = tools.Statistics(lambda ind: ind.fitness.values)  
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, stats=stats, halloffame=hof, verbose=True)
    print("=== Final Results ===")
    print(log)

    print("Melhor Indivíduo:")
    print(hof[0])


if __name__ == "__main__":
    main()
