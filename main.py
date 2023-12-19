import pandas as pd
from algorithm import *
import util
from config import *
from util import plot_tour
from util import read_tsp_file
def run():

    # Loading Coordinates # Berlin 52 (Minimum Distance = 7544.3659)
    if 'Coordinates' in FILE_NAME:
        coordinates = pd.read_csv(FILE_NAME, sep='\t')
        coordinates = coordinates.values
    elif 'tsp' in FILE_NAME:
        coordinates = read_tsp_file(FILE_NAME)

    # Obtaining the Distance Matrix
    distance_matrix = util.build_distance_matrix(coordinates)


    if ALG == Alg.cheapest_insertion:
        route, distance = cheapest_insertion(distance_matrix, **PARAMETERS)
    elif ALG == Alg.christofides_algorithm:
        route, distance = christofides_algorithm(distance_matrix, local_search=True, verbose=True)
    elif ALG == Alg.farthest_insertion:
        route, distance = farthest_insertion(distance_matrix, **PARAMETERS)
    elif ALG == Alg.genetic_algorithm:
        route, distance = genetic_algorithm(distance_matrix, **PARAMETERS)
    elif ALG == Alg.greedy_karp_steele_patching:
        route, distance = greedy_karp_steele_patching(distance_matrix, **PARAMETERS)
    elif ALG == Alg.lkh:
        import requests
        import lkh
        problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
        problem = lkh.LKHProblem.parse(problem_str)
        solver_path = 'LKH-3.0.6/LKH_mac'
        lkh.solve(solver_path, problem=problem, **PARAMETERS)
    elif ALG == Alg.local_search_2_opt:
        seed = util.seed_function(distance_matrix)
        route, distance = local_search_2_opt(distance_matrix, seed, **PARAMETERS)
    elif ALG == Alg.local_search_3_opt:
        seed = util.seed_function(distance_matrix)
        route, distance = local_search_3_opt(distance_matrix, seed, **PARAMETERS)
    elif ALG == Alg.nearest_insertion:
        route, distance = nearest_neighbour(distance_matrix, **PARAMETERS)
    elif ALG == Alg.nearest_neighbour:
        route, distance = nearest_neighbour(distance_matrix, **PARAMETERS)
    elif ALG == Alg.simulated_annealing:
        route, distance = simulated_annealing_tsp(distance_matrix, **PARAMETERS)
    elif ALG == Alg.tabu_search:
        seed = util.seed_function(distance_matrix)
        route, distance = tabu_search(distance_matrix, seed, **PARAMETERS)

    if ALG != Alg.lkh:
        # Plot Locations and Tour
        print('Total Distance: ', round(distance, 2))
        plot_tour(coordinates, city_tour=route, view='notebook', size=10)


def main():
    run()
    pass

if __name__ == '__main__':
    main()