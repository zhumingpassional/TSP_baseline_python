from enum import Enum

FILE_NAME = 'https://github.com/Valdecy/Datasets/raw/master/Combinatorial/TSP-02-Coordinates.txt'


class Alg(Enum):
    _2_opt = '_2_opt'
    _3_opt = '_3_opt'
    cheapest_insertion = 'cheapest_insertion'
    christofides_algorithm = 'christofides_algorithm'
    farthest_insertion = 'farthest_insertion'
    genetic_algorithm = 'genetic_algorithm'
    nearest_insertion = 'nearest_insertion'
    nearest_neighbour = 'nearest_neighbour'
    simulated_annealing = 'simulated_annealing'
    tabu_search = 'tabu_search'
    greedy_karp_steele_patching = 'greedy_karp_steele_patching'


ALG = Alg.farthest_insertion

# Parameters
if ALG == Alg._2_opt:
    PARAMETERS = {
                'recursive_seeding': -1, # Total Number of Iterations. If This Value is Negative Then the Algorithm Only Stops When Convergence is Reached
                'verbose': True
                 }
elif ALG == Alg._3_opt:
    PARAMETERS = {
                'recursive_seeding': -1, # Total Number of Iterations. If This Value is Negative Then the Algorithm Only Stops When Convergence is Reached
                'verbose': True
                 }
elif ALG == Alg.cheapest_insertion:
    PARAMETERS = {
        'verbose': True
    }
elif ALG == Alg.farthest_insertion:
    PARAMETERS = {
        'initial_location': -1,  # -1 =  Try All Locations.
        'verbose': True
    }
elif ALG == Alg.genetic_algorithm:
    PARAMETERS = {
        'population_size': 15,
        'elite': 1,
        'mutation_rate': 0.1,
        'mutation_search': 8,
        'generations': 1000,
        'verbose': True
    }
elif ALG == Alg.greedy_karp_steele_patching:
    PARAMETERS = {
        'verbose': True
    }
elif ALG == Alg.nearest_insertion:
    PARAMETERS = {
        'initial_location': -1,  # -1 =  Try All Locations.
        'verbose': True
    }
elif ALG == Alg.nearest_neighbour:
    PARAMETERS = {
        'initial_location': -1,  # -1 =  Try All Locations.
        'local_search': True,
        'verbose': True
    }
elif ALG == Alg.simulated_annealing:
    PARAMETERS = {
        'initial_temperature': 1.0,
        'temperature_iterations': 10,
        'final_temperature': 0.0001,
        'alpha': 0.9,
        'verbose': True
    }
elif ALG == Alg.tabu_search:
    PARAMETERS = {
        'iterations': 500,
        'tabu_tenure': 75,
        'verbose': True
    }

