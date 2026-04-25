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
    uncertainty_interval              = individual[0:5]
    distance_interval                 = individual[5:10]
    one_cell_priority_interval        = individual[10:15]
    sum_of_priorities_interval        = individual[15:20]
    distance_between_targets_interval = individual[20:25]
    two_cells_priority_interval       = individual[25:30]

    if any(uncertainty_interval[i] < 0 for i in range(4)):
        return False
    if sum(uncertainty_interval) > 2:
        return False

    if any(distance_interval[i] < 0 for i in range(4)):
        return False
    if sum(distance_interval) > 300:
        return False

    if any(one_cell_priority_interval[i] < 0 for i in range(4)):
        return False
    if sum(one_cell_priority_interval) > 1:
        return False

    if any(sum_of_priorities_interval[i] < 0 for i in range(4)):
        return False
    if sum(sum_of_priorities_interval) > 3:
        return False

    if any(distance_between_targets_interval[i] < 0 for i in range(4)):
        return False
    if sum(distance_between_targets_interval) > 400:
        return False

    if any(two_cells_priority_interval[i] < 0 for i in range(4)):
        return False
    if sum(two_cells_priority_interval) > 1:
        return False

    return True


def distance(individual):
    uncertainty_interval              = individual[0:5]
    distance_interval                 = individual[5:10]
    one_cell_priority_interval        = individual[10:15]
    sum_of_priorities_interval        = individual[15:20]
    distance_between_targets_interval = individual[20:25]
    two_cells_priority_interval       = individual[25:30]

    dist1 = dist2 = dist3 = dist4 = dist5 = dist6 = 0.0

    for i in range(4):  # check all but last (last is shaped by the sum constraint)
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

    if sum(uncertainty_interval) > 2:
        dist1 += sum(uncertainty_interval) - 2

    if sum(distance_interval) > 300:
        dist2 += sum(distance_interval) - 300

    if sum(one_cell_priority_interval) > 1:
        dist3 += sum(one_cell_priority_interval) - 1

    if sum(sum_of_priorities_interval) > 3:
        dist4 += sum(sum_of_priorities_interval) - 3

    if sum(distance_between_targets_interval) > 400:
        dist5 += sum(distance_between_targets_interval) - 400

    if sum(two_cells_priority_interval) > 1:
        dist6 += sum(two_cells_priority_interval) - 1

    return 1000 * (dist1/2 + dist2/300 + dist3/1 + dist4/2 + dist5/300 + dist6/1)


def init_individual(icls, generators):
    flat_list = []
    for func in generators:
        flat_list.extend(func())
    return icls(flat_list)


def random_uncertainty_interval():
    hand_tunned = [0.3, 0.3, 0.3, 0.3, 0.3] 
    sigma = 0.15
    return np.random.normal(loc=hand_tunned, scale=sigma)

def random_distance_interval():
    hand_tunned = [50, 50, 50, 50, 50]      
    sigma = 15
    return np.random.normal(loc=hand_tunned, scale=sigma)

def random_one_cell_priority_interval():
    hand_tunned = [0.15, 0.15, 0.15, 0.15, 0.15]   
    sigma = 0.1
    return np.random.normal(loc=hand_tunned, scale=sigma)

def random_sum_of_priorities_interval():
    hand_tunned = [0.5, 0.5, 0.5, 0.5, 0.5] 
    sigma = 0.15
    return np.random.normal(loc=hand_tunned, scale=sigma)

def random_distance_between_targets_interval():
    hand_tunned = [50, 50, 50, 50, 50]
    sigma = 15
    return np.random.normal(loc=hand_tunned, scale=sigma)

def random_two_cells_priority_interval():
    hand_tunned = [0.15, 0.15, 0.15, 0.15, 0.15]   
    sigma = 0.1
    return np.random.normal(loc=hand_tunned, scale=sigma)

def random_rules(size=50):                      # 25 (first system) + 25 (second system)
    return np.random.randint(low=0, high=5, size=size)


###### CUSTOM MUTATION FUNCTION ######
def custom_mutation(individual, indpb):
    # --- Index boundaries (cumulative) ---
    UNCERT_END      = 5   # uncertainty_interval:              [0:5]
    DIST_END        = 10  # distance_interval:                 [5:10]
    OCP_END         = 15  # one_cell_priority_interval:        [10:15]
    SOP_END         = 20  # sum_of_priorities_interval:        [15:20]
    DBT_END         = 25  # distance_between_targets_interval: [20:25]
    TCP_END         = 30  # two_cells_priority_interval:       [25:30]
    RULES_END       = 80  # rules (50 ints):                   [30:80]

    # --- Continuous interval genes ---
    tools.mutGaussian(individual[0         : UNCERT_END], mu=0, sigma=0.20,  indpb=indpb)
    tools.mutGaussian(individual[UNCERT_END : DIST_END],  mu=0, sigma=30.0,  indpb=indpb)
    tools.mutGaussian(individual[DIST_END   : OCP_END],   mu=0, sigma=0.10,  indpb=indpb)
    tools.mutGaussian(individual[OCP_END    : SOP_END],   mu=0, sigma=0.25,  indpb=indpb)
    tools.mutGaussian(individual[SOP_END    : DBT_END],   mu=0, sigma=30.0,  indpb=indpb)
    tools.mutGaussian(individual[DBT_END    : TCP_END],   mu=0, sigma=0.10,  indpb=indpb)

    # --- Integer rule genes (uniform random replacement) ---
    for i in range(TCP_END, RULES_END):
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
        random_sum_of_priorities_interval,
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
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.05, ngen=50, 
                                       stats=stats, halloffame=hof, verbose=True)
    finally:
        pool.close() 
        pool.join()

    print("=== Final Results ===")
    print(log)

    log_filename = "definitive_system/coordination_only/3_drones/tune_more_rules/tune/save.txt"
    with open(log_filename, "w") as f:
        # Use str(log) to get the Logbook content as a string
        f.write(str(log))

    print("=== Top 1 Best Individuals ===")
    for rank, individual in enumerate(hof):
        print(f"Rank {rank + 1}:")
        print(f"Fitness: {individual.fitness.values[0]}")
        print(f"Parameters: {individual}\n")


if __name__ == "__main__":
    main()
