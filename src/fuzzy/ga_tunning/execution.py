import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .fuzzy_inteligent_mobility_protocol import PointOfInterest, drone_protocol_factory
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
    builder.add_handler(VisualizationHandler())
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
        fuzzy_parameters=np.array(individual),
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

def is_feasible(individual):
    uncertainty_interval = individual[0:3]
    distance_interval = individual[3:6]
    individual_cell_uncertainty_interval = individual[6:9]
    one_cell_priority_interval = individual[9:12]

    if uncertainty_interval[0] < 0 or uncertainty_interval[1] < 0 or uncertainty_interval[2] < 0:
        return False
    if uncertainty_interval[0] + uncertainty_interval[1] + uncertainty_interval[2] > 20:
        return False
    
    if distance_interval[0] < 0 or distance_interval[1] < 0 or distance_interval[2] < 0:
        return False
    if distance_interval[0] + distance_interval[1] + distance_interval[2] > 150:
        return False
    
    if individual_cell_uncertainty_interval[0] < 0 or individual_cell_uncertainty_interval[1] < 0 or individual_cell_uncertainty_interval[2] < 0:
        return False
    if individual_cell_uncertainty_interval[0] + individual_cell_uncertainty_interval[1] + individual_cell_uncertainty_interval[2] > 2:
        return False
    
    if one_cell_priority_interval[0] < 0 or one_cell_priority_interval[1] < 0 or one_cell_priority_interval[2] < 0:
        return False
    if one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2] > 1:
        return False

    return True

def distance(individual):
    uncertainty_interval = individual[0:3]
    distance_interval = individual[3:6]
    individual_cell_uncertainty_interval = individual[6:9]
    one_cell_priority_interval = individual[9:12]

    dist1 = 0
    dist2 = 0
    dist3 = 0
    dist4 = 0

    for i in range(2):
        if uncertainty_interval[i] < 0:
            dist1 += abs(uncertainty_interval[i])
        if distance_interval[i] < 0:
            dist2 += abs(distance_interval[i])
        if individual_cell_uncertainty_interval[i] < 0:
            dist3 += abs(individual_cell_uncertainty_interval[i])
        if one_cell_priority_interval[i] < 0:
            dist4 += abs(one_cell_priority_interval[i])
    
    if uncertainty_interval[0] + uncertainty_interval[1] + uncertainty_interval[2] > 20:
        dist1 += (uncertainty_interval[0] + uncertainty_interval[1] + uncertainty_interval[2]) - 20
    if distance_interval[0] + distance_interval[1] + distance_interval[2] > 150:
        dist2 += (distance_interval[0] + distance_interval[1] + distance_interval[2]) - 150
    if individual_cell_uncertainty_interval[0] + individual_cell_uncertainty_interval[1] + individual_cell_uncertainty_interval[2] > 2:
        dist3 += (individual_cell_uncertainty_interval[0] + individual_cell_uncertainty_interval[1] + individual_cell_uncertainty_interval[2]) - 2
    if one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2] > 1:
        dist4 += (one_cell_priority_interval[0] + one_cell_priority_interval[1] + one_cell_priority_interval[2]) - 1

    return 1000*(dist1/20 + dist2/150 + dist3/2 + dist4/1)

def init_individual(icls, generators):
    """
    Initializes a flat individual by concatenating the results 
    of multiple generator functions (which return arrays/lists).
    """
    flat_list = []
    for func in generators:
        # func() returns a NumPy array (e.g., [3.0, 7.0, 12.0]).
        # .tolist() converts it to a standard Python list, which is then extended.
        flat_list.extend(func())
    return icls(flat_list)

def random_uncertainty_interval():
    return([random.uniform(0.0,10.0) for _ in range(3)])

def random_distance_interval():
    return ([random.uniform(0.0, 50.0) for _ in range(3)])

def random_individual_cell_uncertainty_interval():
    return ([random.uniform(0.0, 1.0) for _ in range(3)])

def random_one_cell_priority_interval():
    return ([random.uniform(0.0, 5.0) for _ in range(3)]) 


def main():
    ### Defining the GA ###
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) ## Minimize the accomulated uncertainty 
    creator.create("Individual", list,  fitness=creator.FitnessMin) ## individual

    generate_list = [
        random_uncertainty_interval, 
        random_distance_interval, 
        random_individual_cell_uncertainty_interval, 
        random_one_cell_priority_interval
    ]
    
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual, generate_list)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

    toolbox.register("evaluate", objective_function)
    toolbox.decorate("evaluate", tools.DeltaPenalty(is_feasible, 0, distance))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)                            
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
