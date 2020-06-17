from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import geodesic

# Coordinates
c_KLIA_KUL = (2.7456, 101.7072) # 0. Kuala Lumpur International Airport, Kuala Lumpur
c_SHIA_JAK = (-6.1275, 106.6537) # 1. Soekarno-Hatta International Airport, Jakarta
c_SUVA_BKK = (13.6900, 100.7501) # 2. Suvarnabhumi Airport, Bangkok
c_CLK_HKG = (22.3080, 113.9185) # 3. Chek Lap Kok International Airport, Hong Kong
c_TAO_TPE = (25.0797, 121.2342)  # 4. Taoyuan International Airport, Taipei
c_HND_TOK = (35.5494, 139.7798) # 5. Haneda International Airport, Tokyo
c_ICN_KOR = (37.4602, 126.4407) # 6. Incheon International Airport, Seoul
c_BDA_PEK = (39.5098, 116.4105) # 7. Beijing Daxing International Airport, Beijing

def create_data_model():
    # Stores the data for the problem.
    data = {}
    data['distance_matrix'] =   [[0, (geodesic(c_KLIA_KUL, c_SHIA_JAK).km), (geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_KLIA_KUL, c_BDA_PEK).km)], # Kuala Lumpur International Airport, Kuala Lumpur
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
    print('Shortest Distance (approximate): {} km\n'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    shortestPath_nodeName = 'Shortest flight routes for KUL to all cities (node name):\n'
    shortestPath_nodeIndex = 'Shortest flight routes for KUL to all cities (node index):\n'
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

    shortestPath_nodeIndex += '{} (and vice versa)\n'.format(manager.IndexToNode(index))
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

def main():
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

    # routes = get_routes(solution, routing, manager)
    # # Display the routes.
    # for i, route in enumerate(routes):
    #     print('Route', i, route)

if __name__ == '__main__':
    main()