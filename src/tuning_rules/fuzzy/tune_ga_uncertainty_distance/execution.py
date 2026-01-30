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
CORES_TO_USE = 24

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
        number_of_drones=3,
        map_width=10,
        map_height=10,
        fuzzy_tables=[lookup_one_cell, lookup_two_cells],
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
    medium_uncertainty = (total_uncertainty_drone1+total_uncertainty_drone2+total_uncertainty_drone3)/3

    total_distance_draveled_drone1 = results_aggregator[0]['total_distance_traveled']
    total_distance_draveled_drone2 = results_aggregator[1]['total_distance_traveled']
    total_distance_draveled_drone3 = results_aggregator[2]['total_distance_traveled']
    medium_distance = (total_distance_draveled_drone1 + total_distance_draveled_drone2 + total_distance_draveled_drone3)/3

    final_battery_status_drone1 = results_aggregator[0]['final_battery_status']
    final_battery_status_drone2 = results_aggregator[1]['final_battery_status']
    final_battery_status_drone3 = results_aggregator[2]['final_battery_status']
    medium_battery_final_status = (final_battery_status_drone1+final_battery_status_drone2+final_battery_status_drone3)/3

    total_cost = medium_uncertainty*0.1 + 10000*medium_battery_final_status

    print(f"Individual: {individual}")
    print(f"Variable to be minimized: {total_cost}")
    print(f"Avarege battery final status: {medium_battery_final_status}")
    #print(f"Total number of simulations: {how_many_simulations}")
    return total_cost

########### GA part ##########
def objective_function(individual):
    if not is_feasible(individual):
        return 1000000.0 + distance(individual),    
    
    return create_and_run_simulation(individual),

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
    if distance_interval[0] + distance_interval[1] + distance_interval[2] > 150:
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
    if distance_between_targets_interval[0] + distance_between_targets_interval[1] + distance_between_targets_interval[2] > 150:
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

    if distance_interval[0] + distance_interval[1] + distance_interval[2] > 150:
        dist2 += (distance_interval[0] + distance_interval[1] + distance_interval[2]) - 150

    if one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2] > 1:
        dist4 += (one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2]) - 1

    if sum_of_priorities_interval[0] + sum_of_priorities_interval[1] + sum_of_priorities_interval[2] > 2:
        dist5 += (sum_of_priorities_interval[0] + sum_of_priorities_interval[1] + sum_of_priorities_interval[2]) - 2
    
    if distance_between_targets_interval[0] + distance_between_targets_interval[1] + distance_between_targets_interval[2] > 150:
        dist6 += (distance_between_targets_interval[0] + distance_between_targets_interval[1] + distance_between_targets_interval[2]) - 150
    
    if two_cells_priority_interval[0] + two_cells_priority_interval[1] + two_cells_priority_interval[2] > 1:
        dist6 += (two_cells_priority_interval[0] + two_cells_priority_interval[1] + two_cells_priority_interval[2]) - 1

    return 1000*(dist1/2 + dist2/150 + dist3/1 + dist4/2 + dist5/150 + dist6/1)

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
    hand_tunned = [40, 40, 40]
    sigma = 10
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

def custom_mutation(individual, indpb):
    ## the cromossome has floats and also integers
    ## so it's necessary a custom mutation function
    SPLIT_POINT = 18

    # if the number is below the split point
    tools.mutGaussian(individual[:SPLIT_POINT], mu=0, sigma=0.1, indpb=indpb)

    for i in range(SPLIT_POINT, len(individual)):
        if np.random.random() < indpb:
            individual[i] = np.random.randint(0, 5) # Novo valor inteiro
            
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

    pop = toolbox.population(n=100)                            
    hof = tools.HallOfFame(1)                                
    stats = tools.Statistics(lambda ind: ind.fitness.values)  
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1, ngen=100, 
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

    print("Melhor Indivíduo:")
    print(hof[0])


if __name__ == "__main__":
    main()
