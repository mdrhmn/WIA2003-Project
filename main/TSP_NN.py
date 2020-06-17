# TSP_5: NEAREST NEIGHBOUR (WORKING, APPROX. RESULT)

import numpy as np
from pytsp.constants import Constants
from geopy.distance import geodesic


# Coordinate
c_KLIA_KUL = (2.7456, 101.7072) # 0. Kuala Lumpur International Airport, Kuala Lumpur
c_SHIA_JAK = (-6.1275, 106.6537) # 1. Soekarno-Hatta International Airport, Jakarta
c_SUVA_BKK = (13.6900, 100.7501) # 2. Suvarnabhumi Airport, Bangkok
c_CLK_HKG = (22.3080, 113.9185) # 3. Chek Lap Kok International Airport, Hong Kong
c_TAO_TPE = (25.0797, 121.2342)  # 4. Taoyuan International Airport, Taipei
c_HND_TOK = (35.5494, 139.7798) # 5. Haneda International Airport, Tokyo
c_ICN_KOR = (37.4602, 126.4407) # 6. Incheon International Airport, Seoul
c_BDA_PEK = (39.5098, 116.4105) # 7. Beijing Daxing International Airport, Beijing

shortest_distance_NN = 0

def nearest_neighbour_tsp(graph, starting_point = 0):
    """
    Nearest Neighbour TSP algorithm
    Args:
        graph: 2d numpy array
        starting_point: index of starting node
    Returns:
        tour approximated by nearest Neighbour tsp algorithm
    """

    #? Stores total number of nodes
    number_of_nodes = len(graph)

    #? Stores a list of total number of unvisited nodes (originally all nodes)
    unvisited_nodes = list(range(number_of_nodes))

    #? Remove starting node from unvisited nodes list since that's the node where
    #? we will be starting
    unvisited_nodes.remove(starting_point)

    #? Stores list of all visited nodes
    visited_nodes = [starting_point]

    #! Loop will terminate once all nodes have been visited
    while number_of_nodes > len(visited_nodes):

        #? Declare Neighbours variable and call find_neighbours function to obtain 
        #? list of all neighbours for given node
        #* graph[visited_nodes[-1]]) = get 1 row of distance matrix for current node
        #* [visited_nodes[-1]] = get last row of distance matrix (the number of rows will increase as number of visited nodes increases)
        #* np.array = converts list of distance matrix row into NumPy array type
        Neighbours = find_neighbours(
            np.array(graph[visited_nodes[-1]]))

        #? Declare not_visited_neighbours as a list
        #* Declare as set first to use set library built-in functions (flatten)
        #* Get row from Neighbours (Neighbours[0])
        #* flatten() function = copy the Neighbour list and convert to 1D array
        #* intersection(unvisited_nodes) = only include unvisited neighbours in adjacency list
        not_visited_neighbours = list(
            set(Neighbours[0].flatten()).intersection(unvisited_nodes))

        # Pick the one travelling salesman has not been to yet
        try:
            #? Declare next_node variable and call find_next_neighbour to get the closest neighbour from current node
            #* Passes list of unvisited nodes (not_visited_neighbours) and distance matrix (np.array(graph[visited_nodes[-1]]))
            next_node = find_next_neighbour(
                        not_visited_neighbours, np.array(graph[visited_nodes[-1]]))
                        
        except ValueError:
            print("Nearest Neighbour algorithm couldn't find a Neighbour that hasn't been to yet")
            return

        #? Append closest neighbour (next_node) inside visited_nodes list
        visited_nodes.append(next_node)

        #? Remove closest neighbour (next_node) from unvisited_nodes list
        unvisited_nodes.remove(next_node)

        #? Append returning to starting node and get its distance from last node once all nodes have been visited
        #? (len(unvisited_nodes) == 0)
        if (len(unvisited_nodes) == 0):
            #* Append starting node inside visited_nodes list
            visited_nodes.append(0)

            #? Call find_next_neighbour to get the closest neighbour from final node to starting node
            #* Passes adjacency list (not_visited_neighbours) and distance matrix (np.array(graph[visited_nodes[-1]]))
            next_node = find_next_neighbour(
                not_visited_neighbours, np.array(graph[visited_nodes[-1]]))

    #! Return the generated shortest path
    return visited_nodes


def find_neighbours(array_of_edges_from_node):
    """
        Find the list of all Neighbours for given node from adjacency matrix.
        Args:
            array_of_edges_from_node (np.array):
        Returns:
            List [] of all Neighbours for given node
        """
    #? Generate boolean list named mask where array_of_edges_from_node are +ve
    mask = (array_of_edges_from_node > 0)

    #? Returns a NumPy array where condition is True
    # return np.where(np.squeeze(np.asarray(mask)))
    return np.where(mask == True)


def find_next_neighbour(not_visited_neighbours, array_of_edges_from_node):
    """
    Args:
        not_visited_neighbours:
        array_of_edges_from_node:
    Returns:
        Next closest node (index)
    """
    #? Declare global variable called shortest_distance_NN to store final shortest distance
    global shortest_distance_NN

    #? Declare variable called cheapest path and use NumPy np.argmin function to
    #? find the shortest edge among edges adjacent/connected to current node from
    #? the distance matrix (array_of_edges_from_node)
    #! The np.argmin function will return the INDEX of the shortest edge, not its value
    # Last node in visited_nodes is where the traveling salesman is.   

    #* EXAMPLE:
    #* [1124.73334908 1215.02325381 2534.9099345 3239.05732988 5344.04622076 4600.32543472 4334.83404066]
    #* Shortest edge is at index 0, array_of_edges_from_node[0]. (1124.73334908)
    #* Thus, np.argmin will return index 0.
    #* cheapest_path = 0
    cheapest_path = np.argmin(
        array_of_edges_from_node[not_visited_neighbours])

    #? Add the cheapest path value found into shortest_distance_NN
    shortest_distance_NN += array_of_edges_from_node[not_visited_neighbours][cheapest_path]

    #? Return the node with the corresponding cheapest path
    #* EXAMPLE:
    #* cheapest_path = 0
    #* not_visited_neighbours = [1, 2, 3, 4, 5, 6, 7]
    #* Thus, not_visited_neighbours[0] = 1
    return not_visited_neighbours[cheapest_path]

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

dists = read_distances(text_directory)

shortest_route_NN = nearest_neighbour_tsp(dists)

print("NEAREST NEIGHBOUR ALGORITHM:")
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