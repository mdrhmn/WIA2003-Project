from __future__ import print_function
from collections import deque, namedtuple
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from operator import itemgetter, attrgetter
import gmplot
import math
import matplotlib.pyplot as plt
import numpy as np
import string

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
print("PROBLEM 1")
# PROBLEM 1.1:  Get and mark locations of the all the cities Ben plan to visit. ---------------------------------------------------------------------------------

latitude_list = [2.7456, -6.1275, 13.6900, 22.3080, 25.0797, 35.5494, 37.4602, 39.5098] 
longitude_list = [101.7072, 106.6537, 100.7501, 113.9185, 121.2342, 139.7798, 126.4407, 116.4105] 

gmap3 = gmplot.GoogleMapPlotter(2.7456, 101.7072, 13, apikey="AIzaSyBU-vXpPlrm2_UBJYLzHznpvc_hhYORT8I") 
  
# scatter method of map object  
# scatter points on the google map and display the marker
gmap3.scatter(latitude_list, longitude_list, '#FF0000', 
                              size = 100, marker = True) 

gmap3.draw("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/html/map_markers.html")

# PROBLEM 1.3: Suggest a journey for Ben to visit each of the cities once with the least distance travelled. ----------------------------------------------------

# Main algorithm: Branch and Bound
maxsize = float('inf')

# Function to copy temporary solution to the final solution
def copyToFinal(curr_path):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]

# Function to find the minimum edge cost having an end at the vertex i
def firstMin(adj, i):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]

    return min

# Function to find the second minimum edge cost having an end at the vertex i
def secondMin(adj, i):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]

        elif(adj[i][j] <= second and
                adj[i][j] != first):
            second = adj[i][j]

    return second

# Function that takes as arguments:
# curr_bound -> lower bound of the root node
# curr_weight-> stores the weight of the path so far
# level-> current level while moving
# in the search space tree
# curr_path[] -> where the solution is being stored
# which would later be copied to final_path[]
def TSPRec(adj, curr_bound, curr_weight, level, curr_path, visited):
    global final_res

    # base case is when we have reached level N
    # which means we have covered all the nodes once
    if level == N:

        # check if there is an edge from
        # last vertex in path back to the first vertex
        if adj[curr_path[level - 1]][curr_path[0]] != 0:

            # curr_res has the total weight
            # of the solution we got
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res:
                copyToFinal(curr_path)
                final_res = curr_res
        return

    # for any other level iterate for all vertices
    # to build the search space tree recursively
    for i in range(N):

        # Consider next vertex if it is not same
        # (diagonal entry in adjacency matrix and
        # not visited already)
        if (adj[curr_path[level-1]][i] != 0 and
                visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]

            # different computation of curr_bound
            # for level 2 from the other levels
            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) +
                                firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) +
                                firstMin(adj, i)) / 2)

            # curr_bound + curr_weight is the actual lower bound
            # for the node that we have arrived on.
            # If current lower bound < final_res,
            # we need to explore the node further
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True

                # call TSPRec for the next level
                TSPRec(adj, curr_bound, curr_weight,
                       level + 1, curr_path, visited)

            # Else we have to prune the node by resetting
            # all changes to curr_weight and curr_bound
            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp

            # Also reset the visited array
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True

# This function sets up final_path
def TSP(adj):

    # Calculate initial lower bound for the root node
    # using the formula 1/2 * (sum of first min +
    # second min) for all edges. Also initialize the
    # curr_path and visited array
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N

    # Compute initial bound
    for i in range(N):
        curr_bound += (firstMin(adj, i) +
                       secondMin(adj, i))

    # Rounding off the lower bound to an integer
    curr_bound = math.ceil(curr_bound / 2)

    # We start at vertex 1 so the first vertex
    # in curr_path[] is 0
    visited[0] = True
    curr_path[0] = 0

    # Call to TSPRec for curr_weight
    # equal to 0 and level 1
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited)

# PROBLEM 1.2:  Get the distances between these destinations. ---------------------------------------------------------------------------------------------------

# Adjacency matrix for the given graph
graph =     [[0, (geodesic(c_KLIA_KUL, c_SHIA_JAK).km), (geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_KLIA_KUL, c_BDA_PEK).km)], # Kuala Lumpur International Airport, Kuala Lumpur
            [(geodesic(c_KLIA_KUL, c_SHIA_JAK).km), 0, (geodesic(c_SHIA_JAK, c_SUVA_BKK).km), (geodesic(c_SHIA_JAK, c_CLK_HKG).km), (geodesic(c_SHIA_JAK, c_TAO_TPE).km), (geodesic(c_SHIA_JAK, c_HND_TOK).km), (geodesic(c_SHIA_JAK, c_ICN_KOR).km), (geodesic(c_SHIA_JAK, c_BDA_PEK).km)], # Soekarno-Hatta International Airport, Jakarta
            [(geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_SHIA_JAK, c_SUVA_BKK).km), 0, (geodesic(c_SUVA_BKK, c_CLK_HKG).km), (geodesic(c_SUVA_BKK, c_TAO_TPE).km), (geodesic(c_SUVA_BKK, c_HND_TOK).km), (geodesic(c_SUVA_BKK, c_ICN_KOR).km) , (geodesic(c_SUVA_BKK, c_BDA_PEK).km)], # Suvarnabhumi Airport, Bangkok
            [(geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_SHIA_JAK, c_CLK_HKG).km), (geodesic(c_SUVA_BKK, c_CLK_HKG).km), 0, (geodesic(c_CLK_HKG, c_TAO_TPE).km), (geodesic(c_CLK_HKG, c_HND_TOK).km), (geodesic(c_CLK_HKG, c_ICN_KOR).km), (geodesic(c_CLK_HKG, c_BDA_PEK).km)], # Chek Lap Kok International Airport, Hong Kong
            [(geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_SHIA_JAK, c_TAO_TPE).km) , (geodesic(c_SUVA_BKK, c_TAO_TPE).km), (geodesic(c_CLK_HKG, c_TAO_TPE).km), 0, (geodesic(c_TAO_TPE, c_HND_TOK).km), (geodesic(c_TAO_TPE, c_ICN_KOR).km), (geodesic(c_TAO_TPE, c_BDA_PEK).km)], # Taoyuan International Airport, Taipei
            [(geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_SHIA_JAK, c_HND_TOK).km), (geodesic(c_SUVA_BKK, c_HND_TOK).km), (geodesic(c_CLK_HKG, c_HND_TOK).km), (geodesic(c_TAO_TPE, c_HND_TOK).km), 0, (geodesic(c_HND_TOK, c_ICN_KOR).km), (geodesic(c_HND_TOK, c_BDA_PEK).km)], # Haneda International Airport, Tokyo
            [(geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_SHIA_JAK, c_ICN_KOR).km), (geodesic(c_SUVA_BKK, c_ICN_KOR).km), (geodesic(c_CLK_HKG, c_ICN_KOR).km), (geodesic(c_TAO_TPE, c_ICN_KOR).km), (geodesic(c_HND_TOK, c_ICN_KOR).km), 0, (geodesic(c_ICN_KOR, c_BDA_PEK).km)], # Incheon International Airport, Seoul
            [(geodesic(c_KLIA_KUL, c_BDA_PEK).km), (geodesic(c_SHIA_JAK, c_BDA_PEK).km), (geodesic(c_SUVA_BKK, c_BDA_PEK).km) , (geodesic(c_CLK_HKG, c_BDA_PEK).km), (geodesic(c_TAO_TPE, c_BDA_PEK).km), (geodesic(c_HND_TOK, c_BDA_PEK).km), (geodesic(c_ICN_KOR, c_BDA_PEK).km), 0]] # Beijing Daxing International Airport, Beijing

# Number of nodes
N = 8

# final_path[] stores the final solution
# i.e. the // path of the salesman.
final_path = [None] * (N + 1)
cities_list = ["KUL", "JAK", "BKK", "HKG", "TPE", "TOK", "KOR", "PEK"]

# visited[] keeps track of the already
# visited nodes in a particular path
visited = [False] * N

# Stores the final minimum weight of shortest tour.
final_res = maxsize

TSP(graph)

print("\nBRANCH AND BOUND ALGORITHM:")
print("Shortest distance    : {} km".format(final_res))
print("Shortest path (index):", end =' ')
for i in range(N + 1):
    if (i + 1 < N + 1):
        print(final_path[i], end =' -> ')
    else:
        print(final_path[i], end = ' (and vice versa)\n')

print("Shortest path (nodes):", end =' ')
for i in range(N + 1):
    if (i + 1 < N + 1):
        print(cities_list[final_path[i]], end = ' -> ')
    else:
        print(cities_list[final_path[i]], end = ' (and vice versa)\n')

# BACKUP ALGORITHM: GOOGLE'S OR-TOOLS
print("\nVALIDATING SHORTEST DISTANCE AND ROUTE GENERATED:")
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
    print('Shortest Distance (approximate): {} km\n'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    shortestPath_nodeName = 'Shortest flight routes for KLIA to all cities (node name):\n'
    shortestPath_nodeIndex = 'Shortest flight routes for KLIA to all cities (node index):\n'
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

########################################################################### PROBLEM 2 ###########################################################################
print("PROBLEM 2\n")

# PROBLEM 2.5: Filter stops words from the text you found ----------------------------------------------------------------------------------

directory = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/"
filesArr = ["jak.txt","bkk.txt","hkg.txt","tpe.txt","tok.txt","kor.txt","pek.txt"]
stopWordsDirectory = directory + "stopwords.txt"
wordCount_before = [0,0,0,0,0,0,0]
index = 0

# Given a list of words, remove any that are in a list of stop words.
def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

print("STOP WORDS REMOVAL:")
for i in range(0, len(filesArr), 1):
    fileDirectory = directory + filesArr[index]
    
    # Count total words after stop words removal
    totalCount = 0
    with open(fileDirectory, encoding = "utf8" ) as word_list:
        words = word_list.read().lower().split()

    totalCount = len(words)

    print("Total word count of {0} before stop words removal: {1}".format(filesArr[index], totalCount))
    wordCount_before[index] = totalCount

    # # file with stopwords
    f1 = open(stopWordsDirectory, "r+",encoding="utf8") 

    # # city text file
    f2 = open(fileDirectory, "r+", encoding="utf8")

    file1_raw = f1.read()
    file2_raw = f2.read().lower()

    stopwords = file1_raw.split()
    file2_words_SWRemoved = file2_raw.split()

    # Remove punctuations
    punctuations = string.punctuation
    punctuations += '“”‘’—'
    table = str.maketrans('', '', punctuations)
    stripped = [w.translate(table) for w in file2_words_SWRemoved]

    # Remove stop words
    wordlist = removeStopwords(stripped, stopwords)

    # Write edited text file content back
    f2_w = open(fileDirectory, "w", encoding="utf8") 
    f2_w.write((" ").join(wordlist))
    f2_w.close()
    f1.close()
    f2.close()

    # Count total words after stop words removal
    totalCount = 0
    with open(fileDirectory, encoding = "utf8") as word_list:
        words = word_list.read().lower().split()

    totalCount = len(words)
    
    print("Total word count of {0} after stop words removal: {1}\n".format(filesArr[index], totalCount))

    index = index + 1

# PROBLEM 2.7: Compare words in the webpages with the positive, negative and neutral English words using a String Matching algorithm. --------------------------

print("+VE/-VE WORDS ANALYSIS:")

directory = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/"
filesArr = ["jak.txt","bkk.txt","hkg.txt","tpe.txt","tok.txt","kor.txt","pek.txt"]

# Word sets
# JAKARTA: 
jak_pos = ['growth','grow','grew','optimistic','sufficient','highest','higher','expanded','rise','rising','high-income','easing','productive','supporting','supported','equitable','maintain','stable','better','advantage']
jak_neg = ['shave','downturn','fell','slowdown','lowest','risk','fall','weakening','weaken','deficit','worsen','worst','burden','stress','downfall','decline','risky','risks','declining','declined']

# BANGKOK: 
bkk_pos = ['normal', 'maintain', 'strong', 'plan', 'plans' 'positive', 'increased', 'spike', 'rise', 'sufficient', 'stable', 'reliable', 'manageable', 'surged', 'spiked', 'soared', 'rose', 'relief', 'expand', 'accelerated']
bkk_neg = ['risks', 'irregularity', 'falling', 'weak', 'weakest', 'damage', 'damaging', 'reeling', 'collapse', 'drop', 'loss', 'losses', 'slowdown', 'risk', 'negative', 'critical', 'slower', 'fell', 'stop', 'fall']

# HONG KONG: 
hkg_pos = ['rise', 'maintain', 'supporting', 'stability', 'recover', 'bounce', 'restore', 'save', 'fund', 'boost', 'stimulating', 'increase', 'increased', 'rebooting', 'developing', 'develop', 'relieve', 'grow', 'better', 'stimulate']
hkg_neg = ['deteriorating', 'falling', 'worse', 'weakening', 'slowing', 'unemployment', 'threat', 'deficit', 'risk', 'reduced', 'lower', 'weakening', 'crisis', 'recession', 'downturn', 'unrest', 'fell',  'slowing', 'reeling', 'critical']

# TAIPEI: 
tpe_pos = ['ramped', 'authorised', 'growing', 'expand', 'increase', 'sufficient', 'secure', 'high', 'shoring', 'stability', 'revive', 'maintaining', 'soared', 'rise', 'rising', 'increased', 'surged', 'strong',  'maintain', 'revised']
tpe_neg = ['lowered', 'downgrade', 'uncertainty', 'cut', 'crisis', 'canceled', 'fallen', 'losses', 'subsides', 'slow', 'low', 'recession', 'worst', 'interrupt', 'reduced', 'fell', 'dropped', 'plunge', 'lower', 'downgraded']

# TOKYO: 
tok_pos = ['deploy','raising','raised','raises','greater','recovering','expanded','improvement','improve','expansion','expanding','forestall','funded','protected','overcome','helped','upgrades','improved','recouped','recovery']
tok_neg = ['stagnation','crisis','stagnate','risks','recession','losses','hurt','loss','cancellation','worse','risk','conflicts','worsens','delayed','negative','lower','sicker','bad','drop','disruption']

# SEOUL: 
kor_pos = ['recovery','raise','prevention','recover','rebound','improved','efforts','stronger','additional','rose','expansionary','ensure','helped','stabilization','improving','expended','improvement','sufficiently','substantial','gain']
kor_neg = ['suffer','suffered','recession','severe','setbacks','suspended','warns','delaying','delayed','shutdowns','damage','critical','fallout','dropped','problematic','crisis','negatively','hit','worse','falling']

# BEIJING: 
pek_pos = ['growth','recovery','supporting','increasing','stabilizing','maintained','grow','grew','improvement','upwards','boost','upgraded','achieving','stability','increasing','sustainably','expand','recovery','strengthened','positively']
pek_neg = ['decline','collapse','imbalance','dropped','fell','debt','contract','drop','bottom','weak','suffered','stressed','risk','scandals','degradation','tension','declines','difficult','reducing','weaker']

# 2D Array: [posWords, negWords] for all 7 cities
posnegArr = [[jak_pos, jak_neg], [bkk_pos, bkk_neg], [hkg_pos, hkg_neg], [tpe_pos, tpe_neg], [tok_pos, tok_neg], [kor_pos, kor_neg], [pek_pos, pek_neg]]

# 2D Array; [posCount, negCount] for 7 cities
posnegCountArr = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

# Stores the pos/neg percentage for 7 cities
posPercentArr = [0,0,0,0,0,0,0]
negPercentArr = [0,0,0,0,0,0,0]

# Stores total word counts of each city
totalCountArr = [0,0,0,0,0,0,0]

# Positive and negative word frequencies for each city
jak_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
jak_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
bkk_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
bkk_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
hkg_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
hkg_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
tpe_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
tpe_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
tok_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
tok_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
kor_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
kor_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pek_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pek_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# 2D Array; [posFreq, negFreq] for 7 cities
posnegFrequencyArr = [[jak_pos_f, jak_neg_f], [bkk_pos_f, bkk_neg_f], [hkg_pos_f, hkg_neg_f], [tpe_pos_f, tpe_neg_f], [tok_pos_f, tok_neg_f], [kor_pos_f, kor_neg_f], [pek_pos_f, pek_neg_f]]

index = 0

# First loop: Loops between cities text file
for i in range(0, len(filesArr), 1):

    fileDirectory = directory + filesArr[index]

    # Open and read cities text file
    with open(fileDirectory, encoding="utf8") as word_list:
        words = word_list.read().lower().split()
    
    index2 = 0
    # Second loop: Loops between cities' sets of positive and negative words
    for j in posnegArr:

        totalCount = 0
        # Third loop: Compare the words
        for k in words:
            totalCount = totalCount + 1

            if k in posnegArr[index][index2]:
                posnegCountArr[index][index2] = posnegCountArr[index][index2] + 1
                posnegFrequencyArr[index][index2][(posnegArr[index][index2]).index(k)] = posnegFrequencyArr[index][index2][(posnegArr[index][index2]).index(k)] + 1
     
        if (index2 + 1 < 2):
            index2 = index2 + 1
        else:
            break
        
    totalCountArr[index] = totalCount
    print("{}".format(filesArr[index].upper()))
    print("Total number of words in {0}: {1}\n".format(filesArr[index], totalCount))

    print("Number of positive words in {0}: {1}".format(filesArr[index], posnegCountArr[index][0]))
    print("Positive word sets:\n{0}\nFrequency:\n{1}\n".format(posnegArr[index][0], posnegFrequencyArr[index][0]))
    
    print("Number of negative words in {0}: {1}".format(filesArr[index], posnegCountArr[index][1]))
    print("Negative word sets:\n{0}\nFrequency:\n{1}\n".format(posnegArr[index][1], posnegFrequencyArr[index][1]))
   
    posPercent = (posnegCountArr[index][0]/totalCount)*100
    negPercent = (posnegCountArr[index][1]/totalCount)*100

    posPercentArr[index] = posPercent
    negPercentArr[index] = negPercent

    print("Percentage of positive words in {}: {:.4f}%".format(filesArr[index], posPercent))
    print("Percentage of negative words in {}: {:.4f}%\n".format(filesArr[index], negPercent))

    index = index + 1

# PROBLEM 2.6: Plot line/scatter/histogram graphs related to the word count using Plotly (word count, stop words). ---------------------------------------------

totalCount = [totalCountArr[0], totalCountArr[1], totalCountArr[2], totalCountArr[3], totalCountArr[4], totalCountArr[5], totalCountArr[6]]
labels = ('Jakarta', 'Bangkok', 'Hong Kong', 'Taipei', 'Tokyo', 'Seoul', 'Beijing')
y_pos = np.arange(len(labels))

plt.title('Cities Word Counts')
plt.xlabel('Cities')
plt.ylabel('Number of Words')

plt.plot(labels, wordCount_before, '-', label='Before stop words removal')
plt.plot(labels, totalCount, '-', label='After stop words removal')
plt.legend()

plt.scatter(y_pos, wordCount_before, color=['black', 'red', 'green', 'blue', 'cyan','yellow','orange'])
plt.scatter(y_pos, totalCount, color=['black', 'red', 'green', 'blue', 'cyan','yellow','orange'])

for x,y in zip(labels, wordCount_before):

    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords = "offset points", # how to position the text
                 xytext = (0,10), # distance from text to points (x,y)
                 ha = 'center')

for x,y in zip(labels, totalCount):

    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords = "offset points", # how to position the text
                 xytext = (0,10), # distance from text to points (x,y)
                 ha = 'center')
plt.show()

# PROBLEM 2.8: Plot histogram graphs of positive and negative words found in the webpages. ---------------------------------------------------------------------

N = 7
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [posnegCountArr[0][0], posnegCountArr[1][0], posnegCountArr[2][0], posnegCountArr[3][0], posnegCountArr[4][0], posnegCountArr[5][0], posnegCountArr[6][0]]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = [posnegCountArr[0][1], posnegCountArr[1][1], posnegCountArr[2][1], posnegCountArr[3][1], posnegCountArr[4][1], posnegCountArr[5][1], posnegCountArr[6][1]]
rects2 = ax.bar(ind+width, zvals, width, color='r')

ax.set_xlabel('Cities')
ax.set_ylabel('Number of words')
plt.title('Number of Positive & Negative Words Per Cities')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Jakarta', 'Bangkok', 'Hong Kong', 'Taipei', 'Tokyo', 'Seoul', 'Beijing'))
ax.legend((rects1[0], rects2[0]), ('Positive', 'Negative'))

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

# PROBLEM 2.9: Give an algorithmic conclusion regarding the sentiment of those articles. -----------------------------------------------------------------------

def InsertionSort(array):                               # sort values in array in
    for x in range(1, len(array)):                      # x=1 until 5
        for i in range(x, 0, -1):                       # i=-x and decreasing
            if array[i] > array[i - 1]:                 # cause of sorting in ascending
                t = array[i]
                array[i] = array[i - 1]
                array[i - 1] = t
            else:
                break
            i = i - 1
    return array


def Merge(arrA, arrB):
    a = 0
    b = 0
    arrC = []                                       # array for collecting arrA and arrB

    while a < len(arrA) and b < len(arrB):
        if arrA[a] > arrB[b]:                       # causes of array in ascending
            arrC.append(arrA[a])
            a = a + 1
        elif arrA[a] < arrB[b]:                     # causes of array in ascending
            arrC.append(arrB[b])
            b = b + 1
        else:
            arrC.append(arrA[a])
            arrC.append(arrB[b])
            a = a + 1
            b = b + 1

    while a < len(arrA):
        arrC.append(arrA[a])
        a = a + 1

    while b < len(arrB):
        arrC.append(arrB[b])
        b = b + 1

    return arrC

# Using Tim Sort algorithm to sort the sentiments
def TimSort(arr):

    # splitting an array
    for x in range(0, len(arr), RUN):
        arr[x: x + RUN] = InsertionSort(arr[x: x + RUN])  # arr[0 to 2] = InsertionSort(arr[0 to 2])
    RUNinc = RUN                                          # arr[2 to 4] = InsertionSort(arr[2 to 4])
                                                          # arr[4 to 6] = InsertionSort(arr[4 to 6])

    while RUNinc < len(arr):
        for x in range(0, len(arr), 2 * RUNinc):
            arr[x: x + 2 * RUNinc] = Merge(arr[x: x + RUNinc], arr[x + RUNinc: x + 2 * RUNinc])
        RUNinc = RUNinc * 2

# RUN = separate an array into every sub arrays has 2 values
RUN = 2           
cities = ['Jakarta', 'Bangkok', 'Hong Kong', 'Taipei', 'Tokyo', 'Seoul', 'Beijing']

sentimentTemp = [0,0,0,0,0,0,0]
sentimentArr = [0,0,0,0,0,0,0]

index = 0

print("SENTIMENT CONCLUSION:")
print("Concluding sentiments of each cities:\n")
for i in range(len(cities)):
    
    print("For {}:".format(cities[index]))
    print("Positive sentiment: {:.4f}%\tNegative sentiment: {:.4f}%".format(posPercentArr[index], negPercentArr[index]))

    if (posPercentArr[index] > negPercentArr[index]):
        sentimentTemp[index] = posPercentArr[index]
        print("Since {:.4f}% > {:.4f}%, {} has positive sentiment.\n".format(posPercentArr[index], negPercentArr[index], cities[index]))

    elif (posPercentArr[index] < negPercentArr[index]):
        sentimentTemp[index] = -negPercentArr[index]
        print("Since {:.4f}% < {:.4f}%, {} has negative sentiment.\n".format(posPercentArr[index], negPercentArr[index], cities[index]))
    
    index = index + 1

index = 0
for i in range(len(sentimentTemp) and len(cities)):
    sentimentArr[index] = [sentimentTemp[index], cities[index]]
    index = index + 1

TimSort(sentimentArr)

print("Cities sorted according to sentiments (descending order):\n\n{}".format(sentimentArr))
    
########################################################################### PROBLEM 3 ###########################################################################

print("\nPROBLEM 3")

# PROBLEM 3.10: Generate the most optimised route (priority on sentiment, then distance). -----------------------------------------------------------------------

def ArrOpt(city, senti):
    for i in range(0, len(Optimised), 1):
        if Optimised[i] == '':
            Optimised[i] = city
            SOpti[i] = senti
            print('Optimised: ',Optimised,'\n')
            break

def Compare(st, pt, sh):
    while st < len(Sent):
        if Sent[st] == '':
            st = st + 1
            continue

        elif Sent[st] == Shortest[sh]:
            print("[{} vs. {}]\n".format(Sent[st], Shortest[sh]))
            ArrOpt(Sent[st], SSent[st])
            pt = NewPoint(Sent[st])
            sh = NewShort(Sent[st])
            Sent[st] = ''
            st = 0
            continue

        else:
            break

    if Optimised[6] != '':
        print('All cities have been compared!\n')
        return None

    else:
        print("[{} vs. {}]\n".format(Sent[st], Shortest[sh]))
        CalcDist(st, pt, sh)


def CalcDist(st, pt, sh):
    d1 = (geodesic(DSent[st], DShort[pt]).km)
    d2 = (geodesic(DShort[sh], DShort[pt]).km)
    DistDiff = abs((d1 - d2)) / d2 * 100
    print("CHECKING DISTANCE:")
    print("Distance from {} to {}: {} km".format(Shortest[pt], Sent[st], d1))
    print("Distance from {} to {}: {} km".format(Shortest[pt], Shortest[sh], d2))

    if DistDiff > 40:
        print('Distance difference:', DistDiff,'> 40% (Rejected)\n')
        Compare(st + 1,pt,sh)

    else:
        print('Distance difference:', DistDiff, '< 40% (Accepted)\n')
        CalcSent(st,pt,sh)


def CalcSent(st, pt, sh):
    sent1 = SSent[st]
    sent2 = SShort[sh]
    sent3 = abs((sent1 - sent2))/abs(sent1) * 100
    print("CHECKING SENTIMENT:")
    print("Sentiment for {}: {}".format(Sent[st], sent1))
    print("Sentiment for {}: {}".format(Shortest[sh], sent2))

    if sent3 > 2:
        print('Sentiment difference:',sent3,'> 2% (Accepted)\n')
        ArrOpt(Sent[st], SSent[st])
        pt = NewPoint(Sent[st])
        sh = NewShort(Sent[st])
        Sent[st] = ''
        Compare(0, pt, sh)

    else:
        print('Sentiment difference:', sent3, '< 2% (Rejected)\n')
        Compare(st + 1, pt, sh)


def NewPoint(city):
    for i in range(0, len(Shortest), 1):
        if Shortest[i] == city:
            print("Next point city: {} at [{}]".format(Shortest[i], i))
            return i
        else:
            continue


def NewShort(city):
    for i in range(0, len(Shortest), 1):
        if Shortest[i] == city:
            if i == 0 or i ==1:
                print("Next city in shortest route: {} at [{}]\n".format(Shortest[i + 1], (i + 1)))
                return i + 1
            else:
                print("Next city in shortest route: {} at [{}]\n".format(Shortest[i - 1], (i - 1)))
                return i - 1
            break

# Sorted location of Shortest List
DShort = [c_SHIA_JAK, c_CLK_HKG, c_TAO_TPE, c_HND_TOK, c_ICN_KOR, c_BDA_PEK,  c_SUVA_BKK]

# Sorted location of Sentiment List
DSent = [c_SHIA_JAK, c_BDA_PEK, c_HND_TOK, c_SUVA_BKK,  c_CLK_HKG, c_ICN_KOR, c_TAO_TPE]

# Sorted sentiment for Shortest List
SShort = [3.0289330922242312, -1.6802800466744459, -1.9270678920841982, -1.4489953632148378, -1.7669795692987302,
          2.7777777777777777, -1.4534055286406384]

# Sorted sentiment for Sentiment List
SSent = [3.0289330922242312, 2.7777777777777777, -1.4489953632148378, -1.4534055286406384, -1.6802800466744459,
         -1.7669795692987302, -1.9270678920841982]

# Shortest = ['Jakarta', 'Hong Kong', 'Taipei', 'Tokyo', 'Korea', 'Beijing', 'Bangkok']
Shortest = ['JAK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK', 'BKK']

# Sent = ['Jakarta', 'Beijing', 'Tokyo', 'Bangkok', 'Hong Kong', 'Korea', 'Taipei']
Sent = ['JAK', 'PEK', 'TOK', 'BKK', 'HKG', 'KOR', 'TPE']

# Optimised = ['Kuala Lumpur', '', '', '', '', '', '', '', 'Kuala Lumpur']
Optimised = ['KUL', '', '', '', '', '', '', '', 'KUL']

# Sentiment array for optimised route
SOpti = ['0', '', '', '', '', '', '', '', '0']

# shortestTemp = ['Kuala Lumpur', 'Jakarta', 'Hong Kong', 'Taipei', 'Tokyo', 'Korea', 'Beijing', 'Bangkok', 'Kuala Lumpur']
shortestTemp = ['KUL', 'JAK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK', 'BKK', 'KUL']

# sentTemp = ['Kuala Lumpur', 'Jakarta', 'Beijing', 'Tokyo', 'Bangkok', 'Hong Kong', 'Korea', 'Taipei', 'Kuala Lumpur']
sentTemp = ['KUL', 'JAK', 'PEK', 'TOK', 'BKK', 'HKG', 'KOR', 'TPE', 'KUL']

print('\nRoute (sort by sentiments):',sentTemp)
print('Route (shortest distance) :',shortestTemp,'\n')

print("GENERATE OPTIMISED ROUTE:")

Compare(0, 0, 0)
print('Optimised route    :', Optimised)
print('Optimised sentiment:', SOpti)

# PROBLEM 3.10: Generate all 5040 possible routes (sorted based on distance, sentiment). ------------------------------------------------------------------------

# Function to generate permutations of a given list 
def permutation(lst): 

    # If lst is empty then there are no permutations 
    if len(lst) == 0: 
        return [] 

    # If there is only one element in lst then, only one permutation is possible 
    if len(lst) == 1: 
        return [lst] 

    # Find the permutations for lst if there are more than 1 characters 
    l = [] # empty list that will store current permutation 

    # Iterate and calculate the permutation 
    for i in range(len(lst)): 
       m = lst[i] 

       # Extract lst[i] or m from the list.  remLst is remaining list
       remLst = lst[:i] + lst[i+1:] 

       # Generating all permutations where m is first element
       for p in permutation(remLst): 
           l.append([m] + p) 

    return l 

# Function to calculate total distance of given route
def distance(allPath):

    coordinate = [c_KLIA_KUL, c_SHIA_JAK, c_SUVA_BKK, c_CLK_HKG, c_TAO_TPE, c_HND_TOK, c_ICN_KOR, c_BDA_PEK]
    totalDist = 0

    for i,x in enumerate(allPath):
        if i == 0:
            continue
        else:
            totalDist += geodesic(coordinate[allPath[i-1]], coordinate[allPath[i]]).km
    
    return totalDist

def getIndex(x):
	routes = ['JAK', 'PEK', 'TOK', 'BKK', 'HKG', 'KOR', 'TPE']
	index = 0

	for i in routes:
		if x == i:
			return index
		else:
			index +=1

def sentimentMeasure(l1, l2, path, dist):
    # Mismatch sentiment values
    current = []   

    # Original indexes
    original = []

    # Sentiment measure
    sent_measure = []

    # sm_value
    sm_value = []

    # Difference of element indexes
    difference = []

    # Total sentiment measures
    totalSMArr = []*5040
    totalSM = 0

    for x,y in enumerate(l2):

        if l1[x] != y:

            # If sentiment value < 0 and l2.index < l1.index (to the left)
            if y < 0 and l2.index(y) < l1.index(y):
                # sent_measure.append(-2 * (abs(l2.index(y) - l1.index(y))))
                sent_measure.append([y, 2 * (abs(l2.index(y) - l1.index(y)))])
                sm_value.append([y, 2])
                totalSM += 2 * (abs(l2.index(y) - l1.index(y)))
            elif y < 0 and l2.index(y) > l1.index(y):
                # sent_measure.append(-1 * (abs(l2.index(y) - l1.index(y))))
                sent_measure.append([y, 1 * (abs(l2.index(y) - l1.index(y)))])
                sm_value.append([y, 1])
                totalSM += 1 * (abs(l2.index(y) - l1.index(y)))
            elif y > 0 and l2.index(y) < l1.index(y):
                # sent_measure.append(-1 * (abs(l2.index(y) - l1.index(y))))
                sent_measure.append([y, 1 * (abs(l2.index(y) - l1.index(y)))])
                sm_value.append([y, 1])
                totalSM += 1 * (abs(l2.index(y) - l1.index(y)))
            elif y > 0 and l2.index(y) > l1.index(y):
                # sent_measure.append(-2 * (abs(l2.index(y) - l1.index(y))))
                sent_measure.append([y, 2 * (abs(l2.index(y) - l1.index(y)))])
                sm_value.append([y, 2])
                totalSM += 2 * (abs(l2.index(y) - l1.index(y)))

            current.append([y, l2.index(y)])  
            difference.append([y, (abs(l2.index(y) - l1.index(y)))])
            # original.append(l1.index(y))
            original.append([y, l1.index(y)])

        totalSMArr.append(totalSM)

    return(l2, totalSM, path, dist)

# List of cities 
cityName = ['JAK', 'BKK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK']
city = [1,2,3,4,5,6,7]

sentBest = [3.029, 2.778, -1.449, -1.453, -1.680, -1.768, -1.927]
sentimentRoutes = []*5040

path = permutation(city) # Generating all possible path using permutation
pathName = permutation(cityName)

# List of total distance of all routes
distList = list() 
tempArr = list()

file2 = open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/POSSIBLE ROUTES (COMBINED)/routes.txt", "r+", encoding="utf8") # original

file3 = open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/POSSIBLE ROUTES (COMBINED)/sentSorted.txt", "r+", encoding="utf8") # sorted sentiment

file4 = open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/POSSIBLE ROUTES (COMBINED)/distSorted.txt", "r+", encoding="utf8") # sorted distance

# To clear existing contents in the text file
file2.truncate(0) 
file3.truncate(0)
file4.truncate(0)

# Adding KUL as start & end destination, and storing total distance for each route
for p, v  in enumerate(path): 
    pathName[p].insert(0, 'KUL')
    pathName[p].append('KUL')
    path[p].insert(0, 0)
    path[p].append(0)

    sentimentRoutes.append([sentBest[getIndex(pathName[p][1])],
                            sentBest[getIndex(pathName[p][2])],
                            sentBest[getIndex(pathName[p][3])],
                            sentBest[getIndex(pathName[p][4])],
                            sentBest[getIndex(pathName[p][5])],
                            sentBest[getIndex(pathName[p][6])],
                            sentBest[getIndex(pathName[p][7])]])

    distList.append(distance(path[p]))

    file2.write("Path [{0:04d}]\t:\t{1}\nDistance\t:\t{2} km\nSentiment\t:\t{3}\n\n".format(p+1, pathName[p], distList[p], sentimentRoutes[p]))

file2.close()

totalSMArr = []*5040
for i in range(len(sentimentRoutes)):
    totalSMArr.append(sentimentMeasure(sentBest, sentimentRoutes[i], pathName[i], distList[i]))

sortedTotalSMArr = sorted(totalSMArr, key = itemgetter(1, 0), reverse=False)

for i in range(len(sentimentRoutes)):
    file3.write("Path [{0:04d}]\t\t:\t{1}\nDistance\t\t:\t{2} km\nSentiment\t\t:\t{3}\nTotal Sentiment Measure\t:\t{4}\n\n".format(i+1,sortedTotalSMArr[i][2], sortedTotalSMArr[i][3],sortedTotalSMArr[i][0], sortedTotalSMArr[i][1]))

file3.close()

# Temp list will store all sorted routes ascendingly (based on total distance).
temp = list()
temp = sorted(totalSMArr, key=itemgetter(3))

# Write the all sorted routes in a new text file.
for y,z in enumerate(totalSMArr):
    file4.write("Path [{0:04d}]\t\t:\t{1}\nDistance\t\t:\t{2} km\nSentiment\t\t:\t{3}\nTotal Sentiment Measure\t:\t{4}\n\n".format(y+1, temp[y][2], temp[y][3], temp[y][1], temp[y][0]))
 
file4.close()

# PROBLEM 3.10: Generate graph of sentiment over all possible routes.  -----------------------------------------------------------------------------------------

x1 = []
# line 1 points 
for i in range (5040):
    x1.append(i)

y1 = []
for i in range (5040):
    y1.append(totalSMArr[i][1])

plt.xlabel("Possible Routes (5040)")
plt.ylabel("Sentiment Measurers")
plt.title("Sentiments of All Possible Routes")
plt.plot(x1, y1, color='r')
plt.show()

# PROBLEM 3.10: Generate graph of distance over all possible routes.  ------------------------------------------------------------------------------------------

x1 = []
# line 1 points 
for i in range (5040):
    x1.append(i)

y1 = []
for i in range (5040):
    y1.append(temp[i][3])

# Line chart
plt.xlabel("Possible Routes (5040)")
plt.ylabel("Total Distance")
plt.title("Distance of All Possible Routes")
plt.plot(x1, y1)
plt.show()

# PROBLEM 3.10: Generate graph of both distance and sentiment over all possible routes.  -----------------------------------------------------------------------

x1 = []
# line 1 points 
for i in range (5040):
    x1.append(i)

y1 = []
for i in range (5040):
    y1.append(temp[i][3])

# line 2 points 
x2 = []
for i in range (5040):
    x2.append(i)

y2 = []
for i in range (5040):
    y2.append(temp[i][1])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Possible Routes (5040)')
ax1.set_ylabel('Distance', color=color)
ax1.plot(x1, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Sentiment Measure', color=color)  # we already handled the x-label with ax1
ax2.plot(x2, y2, color=color, ls='steps', lw=0.1)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# PROBLEM 3.10: Generate graph of comparison between 3 types of routes.  ---------------------------------------------------------------------------------------

# Data to plot
n_groups = 3
distance = (13875.440442360568, 21422.180950592334, 15724.175657669719)
sentiment = (30, 0, 13)

# Create plot
fig, ax1 = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

ax1.set_xlabel('Possible Routes (5040)')
plt.title('Comparison of 3 Types of Routes')
plt.xticks(index + bar_width/2, ('Shortest Distance', 'Best Sentiment', 'Optimised'))
plt.xlabel('Types of Routes')

color = 'tab:blue'
ax1.set_ylabel('Distance', color=color)  # we already handled the x-label with ax1
rects1 = ax1.bar(index, distance, bar_width,
alpha=opacity,
color='b',
label='Distance (km)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Sentiment Measure', color=color)
rects2 = ax2.bar(index + bar_width, sentiment, bar_width,
alpha=opacity,
color='r',
label='Sentiment Measure')
ax2.tick_params(axis='y', labelcolor=color)

# Label values of each bar
for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x()+rect.get_width()/2., h, '%d'%int(h),
            ha='center', va='bottom')

for rect in rects2:
    h = rect.get_height()
    ax2.text(rect.get_x()+rect.get_width()/2., h, '%d'%int(h),
            ha='center', va='bottom')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()