import matplotlib.pyplot as plt
import numpy as np
from collections import deque, namedtuple
from geopy.distance import geodesic
from gmplot import *
import string

print("PROBLEM 3")

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
            # print("--------------------------------------------------------------------------------------------------------------------------")
            print("[{} vs. {}]\n".format(Sent[st], Shortest[sh]))
            # print(Sent[st], ' vs. ', Shortest[sh], '\n')
            ArrOpt(Sent[st], SSent[st])
            pt = NewPoint(Sent[st])
            sh = NewShort(Sent[st])
            Sent[st] = ''
            st = 0
            continue

        else:
            break

    if Optimised[6]!='':
        print('All cities have been compared!\n')
        return None

    else:
        # print("--------------------------------------------------------------------------------------------------------------------------")
        print("[{} vs. {}]\n".format(Sent[st], Shortest[sh]))
        # print(Sent[st], 'vs ',Shortest[sh],'\n')
        CalcDist(st, pt, sh)


def CalcDist(st, pt, sh):
    d1 = (geodesic(DSent[st], DShort[pt]).km)
    d2 = (geodesic(DShort[sh], DShort[pt]).km)
    DistDiff = abs((d1 - d2)) / d2 * 100
    print("CHECKING DISTANCE:")
    print("Distance from {} to {}: {} km".format(Shortest[pt], Sent[st], d1))
    print("Distance from {} to {}: {} km".format(Shortest[pt], Shortest[sh], d2))
    # print('Distance from ',Shortest[pt],' to ',Sent[st],': ', d1,'km')
    # print('Distance from ', Shortest[pt], ' to ', Shortest[sh], ': ', d2, 'km')

    if DistDiff > 40:
        print('Distance difference:', DistDiff,'> 40% (Rejected)\n')
        Compare(st+1,pt,sh)

    else:
        print('Distance difference:', DistDiff, '< 40% (Accepted)\n')
        CalcSent(st,pt,sh)


def CalcSent(st, pt, sh):
    sent1 = SSent[st]
    sent2 = SShort[sh]
    sent3 = abs((sent1-sent2))/abs(sent1) * 100
    print("CHECKING SENTIMENT:")
    print("Sentiment for {}: {}".format(Sent[st], sent1))
    print("Sentiment for {}: {}".format(Shortest[sh], sent2))
    # print('Sentiment for',Sent[st],': ',sent1)
    # print('Sentiment for', Shortest[sh], ': ', sent2)

    if sent3 > 2:
        print('Sentiment difference:',sent3,'> 2% (Accepted)\n')
        ArrOpt(Sent[st], SSent[st])
        pt = NewPoint(Sent[st])
        sh = NewShort(Sent[st])
        Sent[st] = ''
        Compare(0, pt, sh)

    else:
        print('Sentiment difference:', sent3, '< 2% (Rejected)\n')
        Compare(st+1, pt, sh)


def NewPoint(city):
    for i in range(0, len(Shortest), 1):
        if Shortest[i] == city:
            print("Next point city: {} at [{}]".format(Shortest[i], i))
            # print('Next point city:', Shortest[i], 'at [', i, ']')
            return i
        else:
            continue


def NewShort(city):
    for i in range(0, len(Shortest), 1):
        if Shortest[i] == city:
            if i == 0 or i ==1:
                print("Next city in shortest route: {} at [{}]\n".format(Shortest[i + 1], (i + 1)))
                # print('Next city in shortest route:', Shortest[i+1],'at [', i+1, ']\n')
                return i+1
            # elif i == 1:
            #     return i+1
            else:
                print("Next city in shortest route: {} at [{}]\n".format(Shortest[i - 1], (i - 1)))
                # print('Next city in shortest route:', Shortest[i-1], 'at [',i-1, ']\n')
                return i-1
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