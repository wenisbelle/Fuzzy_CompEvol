import random
import logging
import multiprocessing

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .fuzzy_inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
from .lookup_table_generator import FuzzyLookupTable
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np

how_many_simulations = 0
CORES_TO_USE = 16

#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual):
    ##### Configuring global parameter
    global how_many_simulations
    how_many_simulations +=1
    
    ##### Creating the fuzzy lookup tables
    fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters= np.array(individual)) 
    lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()
    
    ##### Configuring the simulation
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
    NUMBER_OF_DRONES = 7

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

    ##### Getting the results of the simulation #####
    medium_uncertainty = 0
    medium_battery_final_status = 0
    battery_penalty = 0
    for i in range(NUMBER_OF_DRONES):
        medium_uncertainty += results_aggregator[i]['accomulated_uncertainty']/NUMBER_OF_DRONES
        medium_battery_final_status += results_aggregator[i]['final_battery_status']/NUMBER_OF_DRONES
        ##### Giving a penalty if the drone ran out of battery #####
        ##### 3 is the enum status for DEAD #####
        if results_aggregator[i]['drone_status'] == 3:
            battery_penalty = 10000
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
    NUMBER_OF_RUNS_PER_EPISODE = 3
    average_return = 0.0
    for _ in range(NUMBER_OF_RUNS_PER_EPISODE):
        if not is_feasible(individual):
            return 100000.0 + distance(individual),    
        average_return += create_and_run_simulation(individual)/NUMBER_OF_RUNS_PER_EPISODE
    return average_return,

def is_feasible(individual):
    uncertainty_interval = individual[0:3]
    distance_interval = individual[3:6]
    one_cell_priority_interval = individual[6:9]
    sum_of_priorities_interval = individual[9:12]
    distance_between_targets_interval = individual[12:15]
    two_cells_priority_interval = individual[15:18]

    if uncertainty_interval[0] < 0 or uncertainty_interval[1] < 0 or uncertainty_interval[2] < 0:
        return False
    if uncertainty_interval[0] + uncertainty_interval[1] + uncertainty_interval[2] > 2:
        return False
    
    if distance_interval[0] < 0 or distance_interval[1] < 0 or distance_interval[2] < 0:
        return False
    if distance_interval[0] + distance_interval[1] + distance_interval[2] > 300:
        return False    
    
    if one_cell_priority_interval[0] < 0 or one_cell_priority_interval[1] < 0 or one_cell_priority_interval[2] < 0:
        return False
    if one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2] > 1:
        return False
    
    if sum_of_priorities_interval[0] < 0 or sum_of_priorities_interval[1] < 0 or sum_of_priorities_interval[2] < 0:
        return False
    if sum_of_priorities_interval[0] + sum_of_priorities_interval[1] + sum_of_priorities_interval[2] > 2:
        return False
    
    if distance_between_targets_interval[0] < 0 or distance_between_targets_interval[1] < 0 or distance_between_targets_interval[2] < 0:
        return False
    if distance_between_targets_interval[0] + distance_between_targets_interval[1] + distance_between_targets_interval[2] > 300:
        return False   
    
    if two_cells_priority_interval[0] < 0 or two_cells_priority_interval[1] < 0 or two_cells_priority_interval[2] < 0:
        return False
    if two_cells_priority_interval[0] + two_cells_priority_interval[1] + two_cells_priority_interval[2] > 1:
        return False    

    return True

def distance(individual):
    uncertainty_interval = individual[0:3]
    distance_interval = individual[3:6]
    one_cell_priority_interval = individual[6:9]
    sum_of_priorities_interval = individual[9:12]
    distance_between_targets_interval = individual[12:15]
    two_cells_priority_interval = individual[15:18]

    dist1 = 0
    dist2 = 0
    dist3 = 0
    dist4 = 0
    dist5 = 0
    dist6 = 0


    for i in range(2):
        if uncertainty_interval[i] < 0:
            dist1 += abs(uncertainty_interval[i])
        if distance_interval[i] < 0:
            dist2 += abs(distance_interval[i])
        if one_cell_priority_interval[i] < 0:
            dist3 += abs(one_cell_priority_interval[i])
        if sum_of_priorities_interval[i] < 0:
            dist4 += abs(sum_of_priorities_interval[i])
        if distance_between_targets_interval[i] < 0:
            dist5 += abs(distance_between_targets_interval[i])
        if two_cells_priority_interval[i] < 0:
            dist6 += abs(two_cells_priority_interval[i])
    
    if uncertainty_interval[0] + uncertainty_interval[1] + uncertainty_interval[2] > 2:
        dist1 += (uncertainty_interval[0] + uncertainty_interval[1] + uncertainty_interval[2]) - 2

    if distance_interval[0] + distance_interval[1] + distance_interval[2] > 300:
        dist2 += (distance_interval[0] + distance_interval[1] + distance_interval[2]) - 300

    if one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2] > 1:
        dist4 += (one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2]) - 1

    if sum_of_priorities_interval[0] + sum_of_priorities_interval[1] + sum_of_priorities_interval[2] > 2:
        dist5 += (sum_of_priorities_interval[0] + sum_of_priorities_interval[1] + sum_of_priorities_interval[2]) - 2
    
    if distance_between_targets_interval[0] + distance_between_targets_interval[1] + distance_between_targets_interval[2] > 300:
        dist6 += (distance_between_targets_interval[0] + distance_between_targets_interval[1] + distance_between_targets_interval[2]) - 300
    
    if two_cells_priority_interval[0] + two_cells_priority_interval[1] + two_cells_priority_interval[2] > 1:
        dist6 += (two_cells_priority_interval[0] + two_cells_priority_interval[1] + two_cells_priority_interval[2]) - 1

    return 1000*(dist1/2 + dist2/300 + dist3/1 + dist4/2 + dist5/300 + dist6/1)

def init_individual(icls, generators):
    """
    Initializes a flat individual by concatenating the results 
    of multiple generator functions (which return arrays/lists).
    """
    flat_list = []
    for func in generators:
        flat_list.extend(func())
    return icls(flat_list)

def random_uncertainty_interval():
    ## hand tunned value
    hand_tunned = [0.25, 0.25, 0.25]
    sigma = 0.25
    return(np.random.normal(loc=hand_tunned, scale=sigma))

def random_distance_interval():
    hand_tunned = [80, 80, 80]
    sigma = 20
    return(np.random.normal(loc=hand_tunned, scale=sigma))

def random_one_cell_priority_interval():
    hand_tunned = [0.25, 0.25, 0.25]
    sigma = 0.10
    return(np.random.normal(loc=hand_tunned, scale=sigma))

def random_sum_of_priotities_interval():
    ## hand tunned value
    hand_tunned = [0.5, 0.5, 0.5]
    sigma = 0.25
    return(np.random.normal(loc=hand_tunned, scale=sigma))

def random_distance_between_targets_interval():
    hand_tunned = [40, 40, 40]
    sigma = 10
    return(np.random.normal(loc=hand_tunned, scale=sigma))

def random_two_cells_priority_interval():
    hand_tunned = [0.25, 0.25, 0.25]
    sigma = 0.10
    return(np.random.normal(loc=hand_tunned, scale=sigma))

def random_rules(size=18):
    return np.random.randint(low=0, high=5, size=size)

###### CUSTOM MUTATION FUNCTION ###########
###### Implementing a custom mutation function due to the different ranges and 
###### types of each variable. We have some floats and ints and also some numbers with 
###### very different ranges.
def custom_mutation(individual, indpb):
    ## the cromossome has floats and also integers
    ## so it's necessary a custom mutation function that takes in account the type and the universe 
    uncertainty_SPLIT = 3
    distance_SPLIT = 4
    one_cell_priority_SPLIT = 9
    sum_of_priorities_SPLIT = 12
    distance_between_targets_SPLIT = 15
    two_cells_priority_SPLIT = 18
    target_rules_SPLIT = 36
        
    # if the number is below the split point then just use the standard mutGaussian function
    # varying the standard deviation for each interval
    tools.mutGaussian(individual[:uncertainty_SPLIT], mu=0, sigma=0.1, indpb=indpb)
    tools.mutGaussian(individual[uncertainty_SPLIT:distance_SPLIT], mu=0, sigma=20.0, indpb=indpb)
    tools.mutGaussian(individual[distance_SPLIT:one_cell_priority_SPLIT], mu=0, sigma=0.10, indpb=indpb)
    tools.mutGaussian(individual[one_cell_priority_SPLIT:sum_of_priorities_SPLIT], mu=0, sigma=0.25, indpb=indpb)
    tools.mutGaussian(individual[sum_of_priorities_SPLIT:distance_between_targets_SPLIT], mu=0, sigma=20.0, indpb=indpb)
    tools.mutGaussian(individual[distance_between_targets_SPLIT:two_cells_priority_SPLIT], mu=0, sigma=0.10, indpb=indpb)

    for i in range(two_cells_priority_SPLIT, target_rules_SPLIT):
        if np.random.random() < indpb:
            individual[i] = np.random.randint(0, 5)

    return (individual,)

def main():
    ### Defining the GA ###
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) ## Minimize the accomulated uncertainty 
    creator.create("Individual", list,  fitness=creator.FitnessMin) ## individual

    generate_list = [
        random_uncertainty_interval, 
        random_distance_interval, 
        random_one_cell_priority_interval,
        random_sum_of_priotities_interval,
        random_distance_between_targets_interval,
        random_two_cells_priority_interval,
        random_rules
    ]
    
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual, generate_list)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=0.05)
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
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.05, ngen=30, 
                                       stats=stats, halloffame=hof, verbose=True)
    finally:
        pool.close() 
        pool.join()

    print("=== Final Results ===")
    print(log)

    log_filename = "definitive_system/coordination_only/7_drones/tune/ga_logbook.txt"
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
