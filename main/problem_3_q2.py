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
SOpti = [0.0, '', '', '', '', '', '', '', 0.0]

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

result = list()
def printArr(a, n): 
    global result
    z = list()
    for i in range(n): 
        z.append(a[i]) 
    result.append(z)
  
# Function to generate permutations of a given list using Heap's Algorithm
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

# path = permutation(city)
path = heapPermutation(city, len(city), len(city))
pathName = list()

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

# x1 = []
# # line 1 points 
# for i in range (5040):
#     x1.append(i)

# y1 = []
# for i in range (5040):
#     y1.append(temp[i][3])

# # line 2 points 
# x2 = []
# for i in range (5040):
#     x2.append(i)

# y2 = []
# for i in range (5040):
#     y2.append(temp[i][1])

# fig, ax1 = plt.subplots()

# color = 'tab:blue'
# ax1.set_xlabel('Possible Routes (5040)')
# ax1.set_ylabel('Distance', color=color)
# ax1.plot(x1, y1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:red'
# ax2.set_ylabel('Sentiment Measure', color=color)  # we already handled the x-label with ax1
# ax2.plot(x2, y2, color=color, ls='steps', lw=0.1)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

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