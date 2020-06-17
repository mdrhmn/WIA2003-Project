import numpy as np
from geopy.distance import geodesic
from gmplot import *

# Coordinates
c_KLIA_KUL = (2.7456, 101.7072) # 0. Kuala Lumpur International Airport, Kuala Lumpur
c_SHIA_JAK = (-6.1275, 106.6537) # 1. Soekarno-Hatta International Airport, Jakarta
c_SUVA_BKK = (13.6900, 100.7501) # 2. Suvarnabhumi Airport, Bangkok
c_CLK_HKG = (22.3080, 113.9185) # 3. Chek Lap Kok International Airport, Hong Kong
c_TAO_TPE = (25.0797, 121.2342)  # 4. Taoyuan International Airport, Taipei
c_HND_TOK = (35.5494, 139.7798) # 5. Haneda International Airport, Tokyo
c_ICN_KOR = (37.4602, 126.4407) # 6. Incheon International Airport, Seoul
c_BDA_PEK = (39.5098, 116.4105) # 7. Beijing Daxing International Airport, Beijing

# Create list of city coordinates
coords_list = [c_KLIA_KUL, c_SHIA_JAK, c_SUVA_BKK, c_CLK_HKG, c_TAO_TPE, c_HND_TOK, c_ICN_KOR, c_BDA_PEK]

# Distance matrix
# dist_matrix =   [[0, 1165.4167561827298, 1172.0724393813703, 2502.0425532518875, 3210.4510321640914, 5317.682375190713, 4567.311999204081, 4296.74896114658], # Kuala Lumpur International Airport, Kuala Lumpur
#                 [1165.4167561827298, 0, 2286.676187938444, 3243.63923826128, 3796.537384774552, 5770.111908377643, 5244.6409179993725, 5151.5256025977405], # Soekarno-Hatta International Airport, Jakarta
#                 [1172.0724393813703, 2286.676187938444, 0, 1687.9368742238516, 2488.681452051923, 4589.188682359893, 3662.5625635921715, 3246.9650018403004], # Suvarnabhumi Airport, Bangkok
#                 [2502.0425532518875, 3243.63923826128, 1687.9368742238516, 0, 806.6350025250727, 2903.8556027991085, 2065.4315507267406, 1921.7845118841735], # Chek Lap Kok International Airport, Hong Kong
#                 [3210.4510321640914, 3796.537384774552, 2488.681452051923, 806.6350025250727, 0, 2122.1914132355682, 1458.8004562389183, 1662.7344909596593], # Taoyuan International Airport, Taipei
#                 [5317.682375190713, 5770.111908377643, 4589.188682359893, 2903.8556027991085, 2122.1914132355682, 0, 1212.5691583935416, 2105.507105484267], # Haneda International Airport, Tokyo
#                 [4567.311999204081, 5244.6409179993725, 3662.5625635921715, 2065.4315507267406, 1458.8004562389183, 1212.5691583935416, 0, 903.6840252123305], # Incheon International Airport, Seoul
#                 [4296.74896114658, 5151.5256025977405, 3246.9650018403004, 1921.7845118841735, 1662.7344909596593, 2105.507105484267, 903.6840252123305, 0]] # Beijing Daxing International Airport, Beijing

dist_matrix =   [[0, (geodesic(c_KLIA_KUL, c_SHIA_JAK).km), (geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_KLIA_KUL, c_BDA_PEK).km)], # Kuala Lumpur International Airport, Kuala Lumpur
                [(geodesic(c_KLIA_KUL, c_SHIA_JAK).km), 0, (geodesic(c_SHIA_JAK, c_SUVA_BKK).km), (geodesic(c_SHIA_JAK, c_CLK_HKG).km), (geodesic(c_SHIA_JAK, c_TAO_TPE).km), (geodesic(c_SHIA_JAK, c_HND_TOK).km), (geodesic(c_SHIA_JAK, c_ICN_KOR).km), (geodesic(c_SHIA_JAK, c_BDA_PEK).km)], # Soekarno-Hatta International Airport, Jakarta
                [(geodesic(c_KLIA_KUL, c_SUVA_BKK).km), (geodesic(c_SHIA_JAK, c_SUVA_BKK).km), 0, (geodesic(c_SUVA_BKK, c_CLK_HKG).km), (geodesic(c_SUVA_BKK, c_TAO_TPE).km), (geodesic(c_SUVA_BKK, c_HND_TOK).km), (geodesic(c_SUVA_BKK, c_ICN_KOR).km) , (geodesic(c_SUVA_BKK, c_BDA_PEK).km)], # Suvarnabhumi Airport, Bangkok
                [(geodesic(c_KLIA_KUL, c_CLK_HKG).km), (geodesic(c_SHIA_JAK, c_CLK_HKG).km), (geodesic(c_SUVA_BKK, c_CLK_HKG).km), 0, (geodesic(c_CLK_HKG, c_TAO_TPE).km), (geodesic(c_CLK_HKG, c_HND_TOK).km), (geodesic(c_CLK_HKG, c_ICN_KOR).km), (geodesic(c_CLK_HKG, c_BDA_PEK).km)], # Chek Lap Kok International Airport, Hong Kong
                [(geodesic(c_KLIA_KUL, c_TAO_TPE).km), (geodesic(c_SHIA_JAK, c_TAO_TPE).km) , (geodesic(c_SUVA_BKK, c_TAO_TPE).km), (geodesic(c_CLK_HKG, c_TAO_TPE).km), 0, (geodesic(c_TAO_TPE, c_HND_TOK).km), (geodesic(c_TAO_TPE, c_ICN_KOR).km), (geodesic(c_TAO_TPE, c_BDA_PEK).km)], # Taoyuan International Airport, Taipei
                [(geodesic(c_KLIA_KUL, c_HND_TOK).km), (geodesic(c_SHIA_JAK, c_HND_TOK).km), (geodesic(c_SUVA_BKK, c_HND_TOK).km), (geodesic(c_CLK_HKG, c_HND_TOK).km), (geodesic(c_TAO_TPE, c_HND_TOK).km), 0, (geodesic(c_HND_TOK, c_ICN_KOR).km), (geodesic(c_HND_TOK, c_BDA_PEK).km)], # Haneda International Airport, Tokyo
                [(geodesic(c_KLIA_KUL, c_ICN_KOR).km), (geodesic(c_SHIA_JAK, c_ICN_KOR).km), (geodesic(c_SUVA_BKK, c_ICN_KOR).km), (geodesic(c_CLK_HKG, c_ICN_KOR).km), (geodesic(c_TAO_TPE, c_ICN_KOR).km), (geodesic(c_HND_TOK, c_ICN_KOR).km), 0, (geodesic(c_ICN_KOR, c_BDA_PEK).km)], # Incheon International Airport, Seoul
                [(geodesic(c_KLIA_KUL, c_BDA_PEK).km), (geodesic(c_SHIA_JAK, c_BDA_PEK).km), (geodesic(c_SUVA_BKK, c_BDA_PEK).km) , (geodesic(c_CLK_HKG, c_BDA_PEK).km), (geodesic(c_TAO_TPE, c_BDA_PEK).km), (geodesic(c_HND_TOK, c_BDA_PEK).km), (geodesic(c_ICN_KOR, c_BDA_PEK).km), 0]] # Beijing Daxing International Airport, Beijing

# Distance list
dist_list = [# KUL edges
            (0, 1, geodesic(c_KLIA_KUL, c_SHIA_JAK)), (0, 2, geodesic(c_KLIA_KUL, c_SUVA_BKK)), (0, 3, geodesic(c_KLIA_KUL, c_CLK_HKG)), (0, 4, geodesic(c_KLIA_KUL, c_TAO_TPE)),
            (0, 5, geodesic(c_KLIA_KUL, c_HND_TOK)), (0, 6, geodesic(c_KLIA_KUL, c_ICN_KOR)), (0, 7, geodesic(c_KLIA_KUL, c_BDA_PEK)), 
            
            # JAK edges
            (1, 2, geodesic(c_SHIA_JAK, c_SUVA_BKK)), (1, 3, geodesic(c_SHIA_JAK, c_CLK_HKG)), (1, 4, geodesic(c_SHIA_JAK, c_TAO_TPE)), 
            (1, 5, geodesic(c_SHIA_JAK, c_HND_TOK)), (1, 6, geodesic(c_SHIA_JAK, c_ICN_KOR)), (1, 7, geodesic(c_SHIA_JAK, c_BDA_PEK)), 
            
            # BKK edges
            (2, 3, geodesic(c_SUVA_BKK, c_CLK_HKG)), (2, 4, geodesic(c_SUVA_BKK, c_TAO_TPE)), (2, 5, geodesic(c_SUVA_BKK, c_HND_TOK)), 
            (2, 6, geodesic(c_SUVA_BKK, c_ICN_KOR)), (2, 7, geodesic(c_SUVA_BKK, c_BDA_PEK)), 
            
            # HKG edges
            (3, 4, geodesic(c_CLK_HKG, c_TAO_TPE)), (3, 5, geodesic(c_CLK_HKG, c_HND_TOK)), (3, 6, geodesic(c_CLK_HKG, c_ICN_KOR)), 
            (3, 7, geodesic(c_CLK_HKG, c_BDA_PEK)), 
            
            # TPE edges
            (4, 5, geodesic(c_TAO_TPE, c_HND_TOK)), (4, 6, geodesic(c_TAO_TPE, c_ICN_KOR )), (4, 7, geodesic(c_TAO_TPE, c_BDA_PEK)), 
            
            # TOK edges
            (5, 6, geodesic(c_HND_TOK, c_ICN_KOR)), (5, 7, geodesic(c_HND_TOK, c_BDA_PEK)), 
            
            # KOR edges
            (6, 7, geodesic(c_ICN_KOR, c_BDA_PEK))]
