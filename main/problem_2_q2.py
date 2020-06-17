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

########################################################################### PROBLEM 2 ###########################################################################
print("PROBLEM 2\n")

# PROBLEM 2.5: Filter stops words from the text you found ----------------------------------------------------------------------------------

directory = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/"
filesArr = ["jak.txt","bkk.txt","hkg.txt","tpe.txt","tok.txt","kor.txt","pek.txt"]
stopWordsDirectory = directory + "stopwords.txt"
wordCount_before = [0,0,0,0,0,0,0]
wordCount_after = [0,0,0,0,0,0,0]

# Given a list of words, remove any that are in a list of stop words.
def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

print("STOP WORDS REMOVAL:")

index = 0
for i in range(0, len(filesArr), 1):
    fileDirectory = directory + filesArr[index]
    
    # Count total words before stop words removal
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

    wordCount_after[index] = totalCount
    index = index + 1

# PROBLEM 2.6: Plot line/scatter/histogram graphs related to the word count using Plotly (word count, stop words). ---------------------------------------------

# Plot difference in total word counts of every city before and after stop words removal

labels = ('Jakarta', 'Bangkok', 'Hong Kong', 'Taipei', 'Tokyo', 'Seoul', 'Beijing')
y_pos = np.arange(len(labels))

plt.title('Cities Word Counts')
plt.xlabel('Cities')
plt.ylabel('Number of Words')

plt.plot(labels, wordCount_before, '-', label='Before stop words removal')
plt.plot(labels, wordCount_after, '-', label='After stop words removal')
plt.legend()

plt.scatter(y_pos, wordCount_before, color=['black', 'red', 'green', 'blue', 'cyan','yellow','orange'])
plt.scatter(y_pos, wordCount_after, color=['black', 'red', 'green', 'blue', 'cyan','yellow','orange'])

for x,y in zip(labels, wordCount_before):

    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords = "offset points", # how to position the text
                 xytext = (0,10), # distance from text to points (x,y)
                 ha = 'center')

for x,y in zip(labels, wordCount_after):

    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords = "offset points", # how to position the text
                 xytext = (0,10), # distance from text to points (x,y)
                 ha = 'center')
plt.show()

# PROBLEM 2.7: Compare words in the webpages with the positive, negative and neutral English words. --------------------------

print("+VE/-VE WORDS ANALYSIS:")

directory = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/"
filesArr = ["jak.txt","bkk.txt","hkg.txt","tpe.txt","tok.txt","kor.txt","pek.txt"]

# Word sets
# JAKARTA: 
jak_pos = ['growth','grow','grew','optimistic','sufficient','highest','higher','expanded','rise','rising','high-income','easing','productive','supporting','supported','equitable','maintain','stable','better','advantage']
jak_neg = ['shave','downturn','fell','slowdown','lowest','risk','fall','weakening','weaken','deficit','worsen','worst','burden','stress','downfall','decline','risky','risks','declining','declined']

# BANGKOK: 
bkk_pos = ['normal', 'maintain', 'strong', 'plan', 'plans', 'positive', 'increased', 'spike', 'rise', 'sufficient', 'stable', 'reliable', 'manageable', 'surged', 'spiked', 'soared', 'rose', 'relief', 'expand', 'accelerated']
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

# index = cities and its correspending text file
index = 0

# First loop: Loops between cities text file
for i in range(0, len(filesArr), 1):

    fileDirectory = directory + filesArr[index]

    # Open and read cities text file
    with open(fileDirectory, encoding="utf8") as word_list:
        words = word_list.read().lower().split()
    
    # index = pos/neg of each city
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

# PROBLEM 2.8: Plot histogram graphs of positive and negative words found in the webpages. ---------------------------------------------------------------------

# Plot individual positive and negative word counts of every city using bar chart.

jak_pos_set = list()
jak_neg_set = list()
bkk_pos_set = list()
bkk_neg_set = list()
hkg_pos_set = list()
hkg_neg_set = list()
tpe_pos_set = list()
tpe_neg_set = list()
tok_pos_set = list()
tok_neg_set = list()
kor_pos_set = list()
kor_neg_set = list()
pek_pos_set = list()
pek_neg_set = list()

for i in range(20):
    jak_pos_set.append([jak_pos[i], jak_pos_f[i]])
    jak_neg_set.append([jak_neg[i], jak_neg_f[i]])
    bkk_pos_set.append([bkk_pos[i], bkk_pos_f[i]])
    bkk_neg_set.append([bkk_neg[i], bkk_neg_f[i]])
    hkg_pos_set.append([hkg_pos[i], hkg_pos_f[i]])
    hkg_neg_set.append([hkg_neg[i], hkg_neg_f[i]])
    tpe_pos_set.append([tpe_pos[i], tpe_pos_f[i]])
    tpe_neg_set.append([tpe_neg[i], tpe_neg_f[i]])
    tok_pos_set.append([tok_pos[i], tok_pos_f[i]])
    tok_neg_set.append([tok_neg[i], tok_neg_f[i]])
    kor_pos_set.append([kor_pos[i], kor_pos_f[i]])
    kor_neg_set.append([kor_neg[i], kor_neg_f[i]])
    pek_pos_set.append([pek_pos[i], pek_pos_f[i]])
    pek_neg_set.append([pek_neg[i], pek_neg_f[i]])


jak_pos_set.sort(key=itemgetter(1))
jak_neg_set.sort(key=itemgetter(1))
bkk_pos_set.sort(key=itemgetter(1))
bkk_neg_set.sort(key=itemgetter(1))
hkg_pos_set.sort(key=itemgetter(1))
hkg_neg_set.sort(key=itemgetter(1))
tpe_pos_set.sort(key=itemgetter(1))
tpe_neg_set.sort(key=itemgetter(1))
tok_pos_set.sort(key=itemgetter(1))
tok_neg_set.sort(key=itemgetter(1))
kor_pos_set.sort(key=itemgetter(1))
kor_neg_set.sort(key=itemgetter(1))
pek_pos_set.sort(key=itemgetter(1))
pek_neg_set.sort(key=itemgetter(1))

pos_fullSet = [jak_pos_set, bkk_pos_set, hkg_pos_set, tpe_pos_set, tok_pos_set, kor_pos_set, pek_pos_set]

neg_fullSet = [jak_neg_set, bkk_neg_set, hkg_neg_set, tpe_neg_set, tok_neg_set, kor_neg_set, pek_neg_set]

cityName = ["Jakarta", "Bangkok", "Hong Kong", "Taipei", "Tokyo", "Seoul", "Beijing"]

N = 20
ind = np.arange(N)

for i in range(len(cityName)):

    # Positive word count
    plt.title("Positive Word Frequency for {}".format(cityName[i]), fontsize=15)
    plt.xlabel("Frequency of words", fontsize=13)
    plt.ylabel("List of words", fontsize=13)

    valueA = [pos_fullSet[i][0][1], pos_fullSet[i][1][1], pos_fullSet[i][2][1], pos_fullSet[i][3][1],pos_fullSet[i][4][1], 
              pos_fullSet[i][5][1], pos_fullSet[i][6][1], pos_fullSet[i][7][1], pos_fullSet[i][8][1],pos_fullSet[i][9][1],
              pos_fullSet[i][10][1], pos_fullSet[i][11][1], pos_fullSet[i][12][1], pos_fullSet[i][13][1],pos_fullSet[i][14][1], 
              pos_fullSet[i][15][1], pos_fullSet[i][16][1], pos_fullSet[i][17][1], pos_fullSet[i][18][1],pos_fullSet[i][19][1]]

    valueB = [pos_fullSet[i][0][0], pos_fullSet[i][1][0], pos_fullSet[i][2][0], pos_fullSet[i][3][0],pos_fullSet[i][4][0], 
              pos_fullSet[i][5][0], pos_fullSet[i][6][0], pos_fullSet[i][7][0], pos_fullSet[i][8][0],pos_fullSet[i][9][0],
              pos_fullSet[i][10][0], pos_fullSet[i][11][0], pos_fullSet[i][12][0], pos_fullSet[i][13][0],pos_fullSet[i][14][0], 
              pos_fullSet[i][15][0], pos_fullSet[i][16][0], pos_fullSet[i][17][0], pos_fullSet[i][18][0],pos_fullSet[i][19][0]]

    plt.barh(ind, valueA, color='g') # Plot the value
    plt.yticks(ind, valueB)

    for index, value in enumerate(valueA):
        plt.text(value, index, str(value), va='center')

    plt.show()

    # Negative word count
    plt.title("Negative Word Frequency for {}".format(cityName[i]), fontsize=15)
    plt.xlabel("Frequency of words", fontsize=13)
    plt.ylabel("List of words", fontsize=13)

    valueA = [neg_fullSet[i][0][1], neg_fullSet[i][1][1], neg_fullSet[i][2][1], neg_fullSet[i][3][1], neg_fullSet[i][4][1], 
              neg_fullSet[i][5][1], neg_fullSet[i][6][1], neg_fullSet[i][7][1], neg_fullSet[i][8][1], neg_fullSet[i][9][1],
              neg_fullSet[i][10][1], neg_fullSet[i][11][1], neg_fullSet[i][12][1], neg_fullSet[i][13][1], neg_fullSet[i][14][1], 
              neg_fullSet[i][15][1], neg_fullSet[i][16][1], neg_fullSet[i][17][1], neg_fullSet[i][18][1], neg_fullSet[i][19][1]]

    valueB = [neg_fullSet[i][0][0], neg_fullSet[i][1][0], neg_fullSet[i][2][0], neg_fullSet[i][3][0], neg_fullSet[i][4][0], 
              neg_fullSet[i][5][0], neg_fullSet[i][6][0], neg_fullSet[i][7][0], neg_fullSet[i][8][0], neg_fullSet[i][9][0],
              neg_fullSet[i][10][0], neg_fullSet[i][11][0], neg_fullSet[i][12][0], neg_fullSet[i][13][0], neg_fullSet[i][14][0], 
              neg_fullSet[i][15][0], neg_fullSet[i][16][0], neg_fullSet[i][17][0], neg_fullSet[i][18][0], neg_fullSet[i][19][0]]

    plt.barh(ind, valueA, color='r') #Plot the value
    plt.yticks(ind, valueB) 

    for index, value in enumerate(valueA): # Loop to label value for each word
        plt.text(value, index, str(value), va='center')

    plt.show()

# Plot total number of positive and negative word counts of every city using bar chart.

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
ax.set_xticks(ind + width/2)
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

def InsertionSort(array):
    for x in range(1, len(array)):
        for i in range(x, 0, -1):
            if array[i] > array[i - 1]:
                t = array[i]
                array[i] = array[i - 1]
                array[i - 1] = t
            else:
                break
            i = i - 1
    return array

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

InsertionSort(sentimentArr)

print("Cities sorted according to sentiments (descending order):\n\n{}".format(sentimentArr))
    