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
            
    print("SentBest   :\t {}".format(l1))
    print("CurrSent   :\t {}".format(l2))
    print("Original   :\t {}".format(original))
    print("Current    :\t {}".format(current))
    print("Difference :\t {}".format(difference))
    # print("SM_Value   :\t {}".format(sm_value))
    # print("SentMeasure:\t {}\n".format(sent_measure))
    print("TotalSM    :\t {}\n".format(totalSM))
    # print("TotalSMArr :\t {}\n".format(totalSMArr))

    return(l2, totalSM, path)

# List of cities
cityName = ['JAK', 'BKK', 'HKG', 'TPE', 'TOK', 'KOR', 'PEK']
city = [1,2,3,4,5,6,7]

sentBest = [3.029, 2.778, -1.449, -1.453, -1.680, -1.768, -1.927]
sentimentRoutes = []*5040

# path = permutation(city) # Generating all possible path using permutation
pathName = permutation(cityName)

file2 = open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/sentiment_routes_sorted.txt", "r+", encoding="utf8") 

# To clear existing contents in the text file
file2.truncate(0) 

# h is index and i is element
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

# this is for sorted dalam group 
# 0 for sentimentRoutes[i],
# 1 is sentBest
sortedTotalSMArr = sorted(totalSMArr, key = itemgetter(1, 0), reverse=False)

for i in range(len(sentimentRoutes)):
    file2.write("Path [{0:04d}]\t\t:\t{1}\nSentiment\t\t:\t{2}\nTotal Sentiment Measure\t:\t{3}\n\n".format(i+1,sortedTotalSMArr[i][2], sortedTotalSMArr[i][0], sortedTotalSMArr[i][1]))

file2.close()

"""
SHORTEST DISTANCE:
Route               : ['KUL', 'BKK', 'PEK', 'KOR', 'TOK', 'TPE', 'HKG', 'JAK', 'KUL']
Distance [0001]     : 13875.440442360568 km
Sentiment [2645]    : -30

BEST SENTIMENT:
Route               : ['KUL', 'JAK', 'PEK', 'TOK', 'BKK', 'HKG', 'KOR', 'TPE', 'KUL']
Distance [1578]     : 21422.180950592334 km
Sentiment [0001]    : 0

MOST OPTIMISED:
Route               : ['KUL', 'JAK', 'BKK', 'PEK', 'KOR', 'TOK', 'HKG', 'TPE', 'KUL']
Distance [0102]     : 15724.175657669719 km
Sentiment [0182]    : -13
"""

# SentBest   :     [3.029, 2.778, -1.449, -1.453, -1.68, -1.768, -1.927]
# CurrSent   :     [2.778, -1.768, -1.449, -1.927, -1.68, -1.453, 3.029]

# [2.778 -> 3.029] 
# if currSent[l2.index] < original[l2.index]: 
# penalise (-1)
# elif 
# [3.029 -> -1.927]

# [-1.449 --] 0
# [-1.68 --] 0 
# [-1.927 -> -1.453]
# [-1.768 -> 2.778]

# Original   :     [[2.778, 1], [-1.768, 5], [-1.927, 6], [-1.453, 3], [3.029, 0]]
# Current    :     [[2.778, 0], [-1.768, 1], [-1.927, 3], [-1.453, 5], [3.029, 6]]
# Difference :     [[2.778, 1], [-1.768, 4], [-1.927, 3], [-1.453, 2], [3.029, 6]]
# TotalSMArr :     [1, 9, 9, 15, 15, 17, 29]
