from geopy.distance import geodesic
from operator import itemgetter, attrgetter

# Function to generate permutations of a given list 
def permutation(lst): 

    # If lst is empty then there are no permutations 
    if len(lst) == 0: 
        return [] 

    # If there is only one element in lst then, only one permuatation is possible 
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

    c_KLIA_KUL = (2.7456, 101.7072) # 0. Kuala Lumpur International Airport, Kuala Lumpur
    c_SHIA_JAK = (-6.1275, 106.6537) # 1. Soekarno-Hatta International Airport, Jakarta
    c_SUVA_BKK = (13.6900, 100.7501) # 2. Suvarnabhumi Airport, Bangkok
    c_CLK_HKG = (22.3080, 113.9185) # 3. Chek Lap Kok International Airport, Hong Kong
    c_TAO_TPE = (25.0797, 121.2342)  # 4. Taoyuan International Airport, Taipei
    c_HND_TOK = (35.5494, 139.7798) # 5. Haneda International Airport, Tokyo
    c_ICN_KOR = (37.4602, 126.4407) # 6. Incheon International Airport, Seoul
    c_BDA_PEK = (39.5098, 116.4105) # 7. Beijing Daxing International Airport, Beijing

    coordinate = [c_KLIA_KUL, c_SHIA_JAK, c_SUVA_BKK, c_CLK_HKG, c_TAO_TPE, c_HND_TOK, c_ICN_KOR, c_BDA_PEK]
    totalDist = 0

    for i,x in enumerate(allPath):
        if i == 0:
            continue
        else:
            totalDist += geodesic(coordinate[allPath[i-1]], coordinate[allPath[i]]).km
    
    return totalDist

# List of cities 
cityName = ['JAK', 'BKK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK']
city = [1,2,3,4,5,6,7]

path = permutation(city) # Generating all possible path using permutation
pathName = permutation(cityName)

# List of total distance of all routes
distList = list() 
tempArr = list()

file2 = open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/distance_routes.txt", "r+", encoding="utf8") 
file3 = open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/distance_routes_Sorted.txt", "r+", encoding="utf8")  
# To clear existing contents in the text file
file2.truncate(0) 
file3.truncate(0)

# Adding KUL as start & end destination, and storing total distance for each route
for p, v  in enumerate(path): 
    pathName[p].insert(0, 'KUL')
    pathName[p].append('KUL')
    path[p].insert(0, 0)
    path[p].append(0)

    distList.append(distance(path[p]))

    file2.write("Path [{0:04d}]\t:\t{1}\nDistance\t:\t{2} km\n\n".format(p+1, pathName[p], distList[p]))

file2.close()

index = 0
for i in range(len(path) and len(distList)):
    tempArr.append([distList[index], pathName[index]])
    index = index + 1

# Temp list will store all sorted routes ascendingly (based on total distance).
temp = list()
temp = sorted(tempArr, key=itemgetter(0))

# Write the all sorted routes in a new text file.
for y,z in enumerate(tempArr):
    file3.write("Path [{0:04d}]\t:\t{1}\nDistance\t:\t{2} km\n\n".format(y+1, temp[y][1], temp[y][0]))

file3.close()