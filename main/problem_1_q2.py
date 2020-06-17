from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from operator import itemgetter, attrgetter
from collections import deque, namedtuple
from pytsp.constants import Constants
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import numpy as np
import itertools
import string
import gmplot
import math

# Coordinates
c_KLIA_KUL = (2.7456, 101.7072) # 0. Kuala Lumpur International Airport, Kuala Lumpur
c_SHIA_JAK = (-6.1275, 106.6537) # 1. Soekarno-Hatta International Airport, Jakarta
c_SUVA_BKK = (13.6900, 100.7501) # 2. Suvarnabhumi Airport, Bangkok
c_CLK_HKG = (22.3080, 113.9185) # 3. Chek Lap Kok International Airport, Hong Kong
c_TAO_TPE = (25.0797, 121.2342)  # 4. Taoyuan International Airport, Taipei
c_HND_TOK = (35.5494, 139.7798) # 5. Haneda International Airport, Tokyo
c_ICN_KOR = (37.4602, 126.4407) # 6. Incheon International Airport, Seoul
c_BDA_PEK = (39.5098, 116.4105) # 7. Beijing Daxing International Airport, Beijing

########################################################################### PROBLEM 1 ###########################################################################

print("PROBLEM 1\n")

# PROBLEM 1.1:  Get and mark locations of the all the cities Ben plan to visit. ---------------------------------------------------------------------------------

latitude_list = [2.7456, -6.1275, 13.6900, 22.3080, 25.0797, 35.5494, 37.4602, 39.5098] 
longitude_list = [101.7072, 106.6537, 100.7501, 113.9185, 121.2342, 139.7798, 126.4407, 116.4105] 

gmap3 = gmplot.GoogleMapPlotter(2.7456, 101.7072, 13, apikey="AIzaSyBU-vXpPlrm2_UBJYLzHznpvc_hhYORT8I") 
  
# scatter method of map object  
# scatter points on the google map and display the marker
gmap3.scatter(latitude_list, longitude_list, '#FF0000', 
                              size = 100, marker = True) 

gmap3.draw("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/html/map_markers.html")


# PROBLEM 1.2:  Get the distances between these destinations. ---------------------------------------------------------------------------------------------------

def calc_distances(filename):
    file1 = open(filename,"w") 

    coordinate = [c_KLIA_KUL, c_SHIA_JAK, c_SUVA_BKK, c_CLK_HKG, c_TAO_TPE, c_HND_TOK, c_ICN_KOR, c_BDA_PEK]
    for i in range (len(coordinate)):
        for j in range (len(coordinate)):
            if j != i:
                if j + 1 != len(coordinate):
                    file1.write(str(geodesic(coordinate[i], coordinate[j]).km))
                    file1.write(' ')
                else:
                    file1.write(str(geodesic(coordinate[i], coordinate[j]).km))  
            else:
                if j + 1 != len(coordinate):
                    file1.write(str(0))
                    file1.write(' ')
                else:
                    file1.write(str(0))
        file1.write('\n')
    file1.close()

arg = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/test/input.txt"
calc_distances(arg)

# PROBLEM 1.3: Suggest a journey for Ben to visit each of the cities once with the least distance travelled. ----------------------------------------------------

# TSP ALGORITHM 1: HELD-KARP ALGORITHM
def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    path.append(0)

    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list((path))

# TSP ALGORITHM 2: NEAREST NEIGHBOUR ALGORITHM

shortest_distance_NN = 0

def nearest_neighbour_tsp(graph, starting_point = 0):
    """
    Nearest Neighbour TSP algorithm
    Args:
        graph: 2d numpy array
        starting_point: index of starting node
    Returns:
        tour approximated by Nearest Neighbour algorithm
    """
    number_of_nodes = len(graph)
    unvisited_nodes = list(range(number_of_nodes))
    unvisited_nodes.remove(starting_point)
    visited_nodes = [starting_point]

    while number_of_nodes > len(visited_nodes):
        Neighbours = find_neighbours(
            np.array(graph[visited_nodes[-1]]))

        not_visited_neighbours = list(
            set(Neighbours[0].flatten()).intersection(unvisited_nodes))
        
        # pick the one travelling salesman has not been to yet
        try:
            next_node = find_next_neighbour(
                not_visited_neighbours, np.array(graph[visited_nodes[-1]]))
        except ValueError:
            print("Nearest Neighbour algorithm couldn't find a Neighbour that hasn't been to yet")
            return
        visited_nodes.append(next_node)
        unvisited_nodes.remove(next_node)

        # Append going back to starting node
        if (len(unvisited_nodes) == 0):
            visited_nodes.append(0)
            next_node = find_next_neighbour(
                not_visited_neighbours, np.array(graph[visited_nodes[-1]]))

    return visited_nodes


def find_neighbours(array_of_edges_from_node):
    """
        Find the list of all Neighbours for given node from adjacency matrix.
        Args:
            array_of_edges_from_node (np.array):
        Returns:
            List [] of all Neighbours for given node
        """
    mask = (array_of_edges_from_node > 0) & (
        array_of_edges_from_node < Constants.MAX_WEIGHT_OF_EDGE)

    return np.where(np.squeeze(np.asarray(mask)))


def find_next_neighbour(not_visited_neighbours, array_of_edges_from_node):
    """
    Args:
        not_visited_neighbours:
        array_of_edges_from_node:
    Returns:
    """
    global shortest_distance_NN

    # last node in visited_nodes is where the traveling salesman is.   
    cheapest_path = np.argmin(
        array_of_edges_from_node[not_visited_neighbours])

    shortest_distance_NN += array_of_edges_from_node[not_visited_neighbours][cheapest_path]
    return not_visited_neighbours[cheapest_path]

# Function to read distance matrix from text file
def read_distances(filename):
    dists = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments
            if line[0] == '#':
                continue

            dists.append(list(map(float, map(str.strip, line.split(' ')))))

    return dists

# Number of nodes
N = 8

cities_list = ["KUL", "JAK", "BKK", "HKG", "TPE", "TOK", "KOR", "PEK"]
text_directory = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/test/input.txt"

# Read distance matrix from text file
dists = read_distances(text_directory)

# Print the distance matrix
print("DISTANCE MATRIX:")
for row in dists:
    print(' '.join([str(n).rjust(3, ' ') for n in row]))

print('')

# Driver codes for Held-Karp Algorithm
shortest_route_dist = held_karp(dists)

print("HELD-KARP ALGORITHM:")
print("Shortest distance    : {} km".format(shortest_route_dist[0]))
print("Shortest path (index):", end=' ')
for i in range(N + 1):
    if (i + 1 < N + 1):
        print(shortest_route_dist[1][i], end=' -> ')
    else:
        print(shortest_route_dist[1][i], end=' (and vice versa)\n')

print("Shortest path (nodes):", end=' ')
for i in range(N + 1):
    if (i + 1 < N + 1):
        print(cities_list[shortest_route_dist[1][i]], end=' -> ')
    else:
        print(cities_list[shortest_route_dist[1][i]],
              end=' (and vice versa)\n')


# Driver codes for Nearest Neighbour Algorithm
shortest_route_NN = nearest_neighbour_tsp(dists)

print("\nNEAREST NEIGHBOUR ALGORITHM:")
print("Shortest distance    : {} km".format(shortest_distance_NN))
print("Shortest path (index):", end=' ')
for i in range(N + 1):
    if (i + 1 < N + 1):
        print(shortest_route_NN[i], end=' -> ')
    else:
        print(shortest_route_NN[i], end=' (and vice versa)\n')

print("Shortest path (nodes):", end=' ')
for i in range(N + 1):
    if (i + 1 < N + 1):
        print(cities_list[shortest_route_NN[i]], end=' -> ')
    else:
        print(cities_list[shortest_route_NN[i]],
              end=' (and vice versa)\n')

error_diff = (abs(shortest_distance_NN - shortest_route_dist[0])/shortest_route_dist[0])  * 100

print("\nThe deviation of shortest distance calculated from Nearest Neighbour Algorithm compared to Held-Karp Algorithm is {:.2f}%.".format(error_diff))
print("Therefore, Held-Karp Algorithm is preferred compared to Nearest Neighbour. (Accuracy > Time Complexity)")

# BACKUP ALGORITHM: GOOGLE'S OR-TOOLS
print("\nVALIDATING SHORTEST DISTANCE AND ROUTE GENERATED:\n")
print("GOOGLE'S OR-TOOLS (VALIDATE RESULT):")
def create_data_model():
    
    # Stores the data for the problem.
    data = {}

    data['distance_matrix']  =  [[0, (geodesic(c_KLIA_KUL, c_SHIA_JAK).km), (geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_KLIA_KUL, c_BDA_PEK).km)], # Kuala Lumpur International Airport, Kuala Lumpur
                                [(geodesic(c_KLIA_KUL, c_SHIA_JAK).km), 0, (geodesic(c_SHIA_JAK, c_SUVA_BKK).km), (geodesic(c_SHIA_JAK, c_CLK_HKG).km), (geodesic(c_SHIA_JAK, c_TAO_TPE).km), (geodesic(c_SHIA_JAK, c_HND_TOK).km), (geodesic(c_SHIA_JAK, c_ICN_KOR).km), (geodesic(c_SHIA_JAK, c_BDA_PEK).km)], # Soekarno-Hatta International Airport, Jakarta
                                [(geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_SHIA_JAK, c_SUVA_BKK).km), 0, (geodesic(c_SUVA_BKK, c_CLK_HKG).km), (geodesic(c_SUVA_BKK, c_TAO_TPE).km), (geodesic(c_SUVA_BKK, c_HND_TOK).km), (geodesic(c_SUVA_BKK, c_ICN_KOR).km) , (geodesic(c_SUVA_BKK, c_BDA_PEK).km)], # Suvarnabhumi Airport, Bangkok
                                [(geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_SHIA_JAK, c_CLK_HKG).km), (geodesic(c_SUVA_BKK, c_CLK_HKG).km), 0, (geodesic(c_CLK_HKG, c_TAO_TPE).km), (geodesic(c_CLK_HKG, c_HND_TOK).km), (geodesic(c_CLK_HKG, c_ICN_KOR).km), (geodesic(c_CLK_HKG, c_BDA_PEK).km)], # Chek Lap Kok International Airport, Hong Kong
                                [(geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_SHIA_JAK, c_TAO_TPE).km) , (geodesic(c_SUVA_BKK, c_TAO_TPE).km), (geodesic(c_CLK_HKG, c_TAO_TPE).km), 0, (geodesic(c_TAO_TPE, c_HND_TOK).km), (geodesic(c_TAO_TPE, c_ICN_KOR).km), (geodesic(c_TAO_TPE, c_BDA_PEK).km)], # Taoyuan International Airport, Taipei
                                [(geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_SHIA_JAK, c_HND_TOK).km), (geodesic(c_SUVA_BKK, c_HND_TOK).km), (geodesic(c_CLK_HKG, c_HND_TOK).km), (geodesic(c_TAO_TPE, c_HND_TOK).km), 0, (geodesic(c_HND_TOK, c_ICN_KOR).km), (geodesic(c_HND_TOK, c_BDA_PEK).km)], # Haneda International Airport, Tokyo
                                [(geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_SHIA_JAK, c_ICN_KOR).km), (geodesic(c_SUVA_BKK, c_ICN_KOR).km), (geodesic(c_CLK_HKG, c_ICN_KOR).km), (geodesic(c_TAO_TPE, c_ICN_KOR).km), (geodesic(c_HND_TOK, c_ICN_KOR).km), 0, (geodesic(c_ICN_KOR, c_BDA_PEK).km)], # Incheon International Airport, Seoul
                                [(geodesic(c_KLIA_KUL, c_BDA_PEK).km), (geodesic(c_SHIA_JAK, c_BDA_PEK).km), (geodesic(c_SUVA_BKK, c_BDA_PEK).km) , (geodesic(c_CLK_HKG, c_BDA_PEK).km), (geodesic(c_TAO_TPE, c_BDA_PEK).km), (geodesic(c_HND_TOK, c_BDA_PEK).km), (geodesic(c_ICN_KOR, c_BDA_PEK).km), 0]] # Beijing Daxing International Airport, Beijing

    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(manager, routing, solution):
    # Prints solution on console.
    cities_list = ["KUL", "JAK", "BKK", "HKG", "TPE", "TOK", "KOR", "PEK"]
    print('Shortest distance    : {} km (approximate)'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    shortestPath_nodeName = 'Shortest path (nodes): '
    shortestPath_nodeIndex = 'Shortest path (index): '
    route_distance = 0
    shortest_distance = ''
    sourceCityIndex = 0
    count = 0

    while not routing.IsEnd(index):
        if (count == 0):
            sourceCityIndex = index
        count = count + 1

        shortestPath_nodeIndex += '{} -> '.format(manager.IndexToNode(index))
        shortestPath_nodeName += '{} -> '.format(cities_list[index])
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    shortestPath_nodeIndex += '{} (and vice versa)'.format(manager.IndexToNode(index))
    shortestPath_nodeName += '{} (and vice versa)\n'.format(cities_list[sourceCityIndex])

    print(shortestPath_nodeIndex)
    print(shortestPath_nodeName)
    # shortest_distance += 'Route distance: {}miles\n'.format(route_distance)

def get_routes(solution, routing, manager):
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.

    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        
        routes.append(route)
    return routes

# Solve the TSP.
data = create_data_model()
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    # Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
solution = routing.SolveWithParameters(search_parameters)

if solution:
    print_solution(manager, routing, solution)

# PROBLEM 1.4: Plot line between the destinations using Google Maps API. -------------------------------------------------------------------

# Plot line (Before)
latitude_list = [2.7456, -6.1275, 13.6900,
                 22.3080, 25.0797, 35.5494, 37.4602, 39.5098]
longitude_list = [101.7072, 106.6537, 100.7501,
                  113.9185, 121.2342, 139.7798, 126.4407, 116.4105]

lat = []*2
lon = []*2

gmap3 = gmplot.GoogleMapPlotter(
    2.7456, 101.7072, 4, apikey='AIzaSyBU-vXpPlrm2_UBJYLzHznpvc_hhYORT8I')

gmap3.scatter(latitude_list, longitude_list, '#FF0000', size=100, marker=True)

for i in range(8):  # n^2
    for j in range(8):
        if i == j:
            continue
        else:
            lat = [latitude_list[i], latitude_list[j]]
            lon = [longitude_list[i], longitude_list[j]]
            gmap3.plot(lat, lon, 'red', edge_width=2)

gmap3.draw(
    "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/html/map_before.html")

# Plot line (After)
latitude_list = [2.7456, -6.1275, 22.3080, 25.0797,
                 35.5494, 37.4602, 39.5098, 13.6900, 2.7456]
longitude_list = [101.7072, 106.6537, 113.9185, 121.2342,
                  139.7798, 126.4407, 116.4105, 100.7501, 101.7072]

gmap2 = gmplot.GoogleMapPlotter(
    2.7456, 101.7072, 4, apikey='AIzaSyBU-vXpPlrm2_UBJYLzHznpvc_hhYORT8I')

gmap2.scatter(latitude_list, longitude_list, '#FF0000', size=100, marker=True)
gmap2.plot(latitude_list, longitude_list, 'red', edge_width=2.5)

gmap3.draw(
    "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/html/map_after.html")
