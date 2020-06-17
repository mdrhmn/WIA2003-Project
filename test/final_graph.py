import matplotlib.pyplot as plt 
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

# Adding KUL as start & end destination, and storing total distance for each route
for p, v  in enumerate(path): 
    pathName[p].insert(0, 'KUL')
    pathName[p].append('KUL')
    path[p].insert(0, 0)
    path[p].append(0)

    distList.append(distance(path[p]))

index = 0
for i in range(len(path) and len(distList)):
    tempArr.append([distList[index], pathName[index]])
    index = index + 1

# Temp list will store all sorted routes ascendingly (based on total distance).
temp = list()
temp = sorted(tempArr, key=itemgetter(0)) 

def getIndex(x):
	routes = ['JAK', 'PEK', 'TOK', 'BKK', 'HKG', 'KOR', 'TPE']
	index = 0

	for i in routes:
		if x == i:
			return index
		else:
			index +=1

def sentimentMeasure(l1, l2, path):
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

    return(l2, totalSM, path)

# List of cities
cityName = ['JAK', 'BKK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK']
city = [1,2,3,4,5,6,7]

sentBest = [3.029, 2.778, -1.449, -1.453, -1.680, -1.768, -1.927]
sentimentRoutes = []*5040

# path = permutation(city) # Generating all possible path using permutation
pathName = permutation(cityName)

for h,i in enumerate(pathName):
    sentimentRoutes.append([sentBest[getIndex(i[0])],
                            sentBest[getIndex(i[1])],
                            sentBest[getIndex(i[2])],
                            sentBest[getIndex(i[3])],
                            sentBest[getIndex(i[4])],
                            sentBest[getIndex(i[5])],
                            sentBest[getIndex(i[6])]])
totalSMArr = []*5040
for i in range(len(sentimentRoutes)):
    totalSMArr.append(sentimentMeasure(sentBest, sentimentRoutes[i], pathName[i]))

sortedTotalSMArr = sorted(totalSMArr, key = itemgetter(1, 0), reverse=False)

# GRAPH ----------------------------------------------------------------------------------
x1 = []
# line 1 points 
for i in range (5040):
    x1.append(i)

y1 = []
for i in range (5040):
    y1.append(temp[i][0])

# line 2 points 
x2 = []
for i in range (5040):
    x2.append(i)

y2 = []
for i in range (5040):
    y2.append(sortedTotalSMArr[i][1])

# # plotting the line 2 points  
# plt.plot(x2, y2, label = "line 2") 
  
# # naming the x axis 
# plt.xlabel('x - axis') 
# # naming the y axis 
# plt.ylabel('y - axis') x
# # giving a title to my graph 
# plt.title('Two lines on same graph!') 
  
# # show a legend on the plot 
# plt.legend() 
  
# # function to show the plot 
# plt.show()

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Possible Routes (5040)')
ax1.set_ylabel('Distance', color=color)
ax1.plot(x1, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Sentiment Measure', color=color)  # we already handled the x-label with ax1
ax2.plot(x2, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

"""
SHORTEST DISTANCE:
Route               : ['KUL', 'BKK', 'PEK', 'KOR', 'TOK', 'TPE', 'HKG', 'JAK', 'KUL']
Sentiment           : [3.029, -1.68, -1.927, -1.449, -1.768, 2.778, -1.453]
Distance [0001]     : 13875.440442360568 km
Sentiment [2645]    : 28

BEST SENTIMENT:       
Route               : ['KUL', 'JAK', 'PEK', 'TOK', 'BKK', 'HKG', 'KOR', 'TPE', 'KUL']
Sentiment           : [3.029, 2.778, -1.449, -1.453, -1.68, -1.768, -1.927]
Distance [1578]     : 21422.180950592334 km
Sentiment [0001]    : 0

MOST OPTIMISED:
Route               : ['KUL', 'JAK', 'BKK', 'PEK', 'KOR', 'TOK', 'HKG', 'TPE', 'KUL']
Sentiment           : [3.029, -1.453, 2.778, -1.768, -1.449, -1.68, -1.927]
Distance [0102]     : 15724.175657669719 km
Sentiment [0182]    : 13
"""