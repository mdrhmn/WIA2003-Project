from geopy.distance import geodesic
from operator import itemgetter, attrgetter

result = list()
def printArr(a, n): 
    global result
    z = list()
    for i in range(n): 
        z.append(a[i]) 
    result.append(z)
  
# Generating permutation using Heap Algorithm 
def heapPermutation(a, size, n): 

    if (size == 1): 
        printArr(a, n) 
        return
  
    for i in range(size): 
        heapPermutation(a,size-1,n)

        if size&1: 
            a[0], a[size-1] = a[size-1],a[0] 
        else: 
            a[i], a[size-1] = a[size-1],a[i] 

    a = result
    result.clear
    return a


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

    for i in range(len(allPath)):
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

# path = permutation(city)
path = heapPermutation(city, len(city), len(city))

pathName = list()

# List of total distance of all routes
distList = list() 
tempArr = list()

file2 = open("/Users/Faidz Hazirah/Documents/SEM 4/ALGO/Project/routes1.txt", "r+", encoding="utf8") # original
# Path [0001]	:	['KUL', 'JAK', 'BKK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK', 'KUL']
# Distance	:	14479.260051271216 km
# Sentiment	:	[3.029, -1.453, -1.68, -1.927, -1.449, -1.768, 2.778]
file3 = open("/Users/Faidz Hazirah/Documents/SEM 4/ALGO/Project/sentSorted1.txt", "r+", encoding="utf8") # sorted sentiment
# Path [0001]		:	['KUL', 'BKK', 'HKG', 'KOR', 'TPE', 'TOK', 'JAK', 'PEK', 'KUL']
# Distance	:	23805.855099866658 km
# Sentiment		:	[-1.453, -1.68, -1.768, -1.927, -1.449, 3.029, 2.778]
# Total Sentiment Measure	:	46
file4 = open("/Users/Faidz Hazirah/Documents/SEM 4/ALGO/Project/distSorted1.txt", "r+", encoding="utf8") # sorted distance

# To clear existing contents in the text file
file2.truncate(0) 
file3.truncate(0)
file4.truncate(0)

# Adding KUL as start & end destination, and storing total distance for each route
for p in range(len(path)): 

    path[p].insert(0, 0)
    path[p].append(0)
   
    pathName.append(['KUL', cityName[path[p][1]-1], cityName[path[p][2]-1], cityName[path[p][3]-1], cityName[path[p][4]-1], cityName[path[p][5]-1], cityName[path[p][6]-1], cityName[path[p][7]-1], 'KUL'])

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

sortedTotalSMArr = sorted(totalSMArr, key = itemgetter(1, 0), reverse=True)

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