import random
import logging
import multiprocessing

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .fuzzy_inteligent_mobility_protocol import drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np

how_many_simulations = 0
CORES_TO_USE = 24

#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual):
    ##### Configuring global parameter
    global how_many_simulations
    how_many_simulations +=1
    
   
    ##### Configuring the simulation
    config = SimulationConfiguration(
        duration=2000, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
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
        distance_norm=individual[0],
        distance_between_drone_norm=individual[1],
        results_aggregator=results_aggregator
    )

    for _ in range(NUMBER_OF_DRONES):
        builder.add_node(ConfiguredDrone, (0, 0, 0))


    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

    ##### Getting the results of the simulation #####
    medium_uncertainty = 0
    medium_battery_final_status = 0
    for i in range(NUMBER_OF_DRONES):
        medium_uncertainty += results_aggregator[i]['accomulated_uncertainty']/NUMBER_OF_DRONES
        medium_battery_final_status += results_aggregator[i]['final_battery_status']/NUMBER_OF_DRONES
        ##### Giving a penalty if the drone ran out of battery #####
        ##### 3 is the enum status for DEAD #####
        if results_aggregator[i]['drone_status'] == 3:
            print(f"Drone ran out of battery")
        
    medium_battery_consumption = 1.0 - medium_battery_final_status
    
    ##### Cost for optimization #####
    total_cost = medium_uncertainty*0.01

    print(f"Individual: {individual}")
    print(f"Variable to be minimized: {total_cost}")
    print(f"Total number of simulations: {how_many_simulations}")
    return total_cost

########### GA part ##########
def objective_function(individual):
    if not is_feasible(individual):
        return 1000000.0,  # Return a large cost for infeasible solutions    
    
    return create_and_run_simulation(individual),

def is_feasible(individual):
    distance_norm=individual[0]
    distance_between_drone_norm=individual[1]

    if distance_norm <= 0:
        return False
    if distance_between_drone_norm <= 0:
        return False
    return True



def main():
    ### Defining the GA ###
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) ## Minimize the accomulated uncertainty 
    creator.create("Individual", list,  fitness=creator.FitnessMin) ## individual
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.1, 5000.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    ### Parallelization
    pool = multiprocessing.Pool(processes=CORES_TO_USE)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=50)                            
    hof = tools.HallOfFame(1)                                
    stats = tools.Statistics(lambda ind: ind.fitness.values)  
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.05, ngen=20, 
                                       stats=stats, halloffame=hof, verbose=True)
    finally:
        pool.close() 
        pool.join()

    print("=== Final Results ===")
    print(log)

    log_filename = "ga_logbook.txt"
    with open(log_filename, "w") as f:
        # Use str(log) to get the Logbook content as a string
        f.write(str(log))

    print("=== Top Best Individuals ===")
    for rank, individual in enumerate(hof):
        print(f"Rank {rank + 1}:")
        print(f"Fitness: {individual.fitness.values[0]}")
        print(f"Parameters: {individual}\n")


if __name__ == "__main__":
    main()


