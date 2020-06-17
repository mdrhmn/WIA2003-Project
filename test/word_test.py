pek_pos = ['growth','recovery','supporting','increasing','stabilizing','maintained','grow','grew','improvement','upwards','boost','upgraded','achieving','stability','increasing','sustainably','expand','recovery','strengthened','positively']
pek_neg = ['decline','collapse','imbalance','dropped','fell','debt','contract','drop','bottom','weak','suffered','stressed','risk','scandals','degradation','tension','declines','difficult','reducing','weaker']
pek_pos_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pek_neg_f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

with open("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/pek.txt", encoding="utf8") as word_list:
    words = word_list.read().lower().split()

pek_posCount = 0
pek_negCount = 0
totalCount_pek = 0

for i in words:
    totalCount_pek = totalCount_pek + 1
    if i in pek_pos:
        pek_posCount = pek_posCount + 1
        pek_pos_f[pek_pos.index(i)] = pek_pos_f[pek_pos.index(i)] + 1

    if i in pek_neg:
        pek_negCount = pek_negCount + 1
        pek_neg_f[pek_neg.index(i)] = pek_neg_f[pek_neg.index(i)] + 1

print("Total word count in pek.txt: {0}".format(totalCount_pek))
print("Total positive word count in pek.txt: {0}\n".format(pek_posCount))
print("List of positive words in pek.txt based on 10 articles:\n")

index = 0
for i in range(len(pek_pos)):  
    strWord = "[" +  pek_pos[index] + "]"
    print("Word: {:<17s}Frequency: {:<10d}\n".format(strWord, pek_pos_f[index]))
    index = index + 1

print("Total negative word count in pek.txt: {0}\n".format(pek_negCount))
print("List of negative words in pek.txt based on 10 articles:\n")

index = 0
for i in range(len(pek_neg)):  
    strWord = "[" +  pek_neg[index] + "]" 
    print("Word: {:<17s}Frequency: {:<10d}\n".format(strWord, pek_neg_f[index]))
    index = index + 1

print("---------------------------------------------------------------------------------------------------------------------")
