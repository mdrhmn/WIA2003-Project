from geopy.distance import geodesic
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

def read_distances(filename):
    file1 = open(filename,"w") 

    coordinate = [c_KLIA_KUL, c_SHIA_JAK, c_SUVA_BKK, c_CLK_HKG, c_TAO_TPE, c_HND_TOK, c_ICN_KOR, c_BDA_PEK]
    for i in range (len(coordinate)):
        for j in range (len(coordinate)):
            if j != i:
                if j + 1 != len(coordinate):
                    file1.write(str(geodesic(coordinate[i], coordinate[j]).km))
                    file1.write(' ')
                else:
                    file1.write(str(geodesic(coordinate[i], coordinate[j]).km))  
            else:
                if j + 1 != len(coordinate):
                    file1.write(str(0))
                    file1.write(' ')
                else:
                    file1.write(str(0))
        file1.write('\n')
    file1.close()

arg = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/test/input.txt"
read_distances(arg)