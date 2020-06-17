import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter, attrgetter

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

# Positive and negative word frequencies for each city
jak_pos_f = [33, 4, 5, 3, 2, 0, 1, 2, 2, 2, 0, 4, 0, 0, 3, 1, 1, 0, 3, 1]
jak_neg_f = [0, 1, 3, 3, 2, 3, 1, 5, 1, 5, 1, 3, 1, 2, 1, 1, 0, 6, 1, 0]
bkk_pos_f = [2, 0, 4, 3, 0, 3, 2, 4, 1, 1, 0, 0, 3, 2, 2, 2, 2, 3, 2, 0]
bkk_neg_f = [6, 1, 3, 1, 2, 2, 1, 1, 1, 3, 8, 2, 4, 6, 2, 1, 1, 4, 0, 2]
hkg_pos_f = [3, 2, 3, 1, 1, 3, 0, 2, 2, 4, 2, 6, 3, 2, 0, 2, 1, 2, 0, 2]
hkg_neg_f = [2, 2, 2, 2, 2, 6, 2, 13, 7, 3, 4, 0, 2, 8, 4, 8, 3, 0, 1, 1]
tpe_pos_f = [1, 0, 2, 4, 10, 1, 2, 4, 2, 2, 1, 1, 1, 4, 1, 2, 1, 6, 3, 1]
tpe_neg_f = [6, 5, 4, 10, 3, 4, 1, 5, 1, 2, 5, 2, 3, 1, 1, 4, 1, 2, 3, 2]
tok_pos_f = [2, 2, 5, 1, 3, 1, 3, 2, 1, 4, 4, 1, 2, 1, 3, 3, 1, 3, 1, 8]
tok_neg_f = [1, 10, 1, 3, 5, 4, 4, 5, 6, 6, 2, 2, 1, 4, 5, 3, 2, 3, 3, 5]
kor_pos_f = [10, 2, 2, 3, 4, 2, 4, 2, 5, 3, 2, 2, 4, 1, 2, 0, 2, 1, 1, 1]
kor_neg_f = [1, 3, 10, 2, 3, 1, 1, 1, 1, 1, 5, 3, 7, 4, 1, 10, 1, 8, 0, 1]
pek_pos_f = [68, 10, 1, 5, 2, 0, 6, 9, 1, 1, 5, 1, 5, 8, 0, 1, 2, 0, 1, 1]
pek_neg_f = [9, 3, 0, 3, 9, 6, 2, 3, 2, 5, 1, 1, 5, 1, 1, 3, 1, 1, 3, 2]

#########################################################################################################################################################################
# info var etc from above already existed in full codes,                                                                                                                #
#########################################################################################################################################################################

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
    ####POSITIVE####
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

    ####NEGATIVE####
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

