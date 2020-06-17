# import gmplot package
import gmplot

# c_KLIA_KUL = (2.7456, 101.7072) # 0. Kuala Lumpur International Airport, Kuala Lumpur
# c_SHIA_JAK = (-6.1275, 106.6537) # 1. Soekarno-Hatta International Airport, Jakarta
# c_SUVA_BKK = (13.6900, 100.7501) # 2. Suvarnabhumi Airport, Bangkok
# c_CLK_HKG = (22.3080, 113.9185) # 3. Chek Lap Kok International Airport, Hong Kong
# c_TAO_TPE = (25.0797, 121.2342)  # 4. Taoyuan International Airport, Taipei
# c_HND_TOK = (35.5494, 139.7798) # 5. Haneda International Airport, Tokyo
# c_ICN_KOR = (37.4602, 126.4407) # 6. Incheon International Airport, Seoul
# c_BDA_PEK = (39.5098, 116.4105) # 7. Beijing Daxing International Airport, Beijing

#################################################BEFORE#########################################

latitude_list = [2.7456, -6.1275, 13.6900,
                 22.3080, 25.0797, 35.5494, 37.4602, 39.5098]
longitude_list = [101.7072, 106.6537, 100.7501,
                  113.9185, 121.2342, 139.7798, 126.4407, 116.4105]

lat = []*2
lon = []*2

gmap3 = gmplot.GoogleMapPlotter(
    2.7456, 101.7072, 4, apikey='AIzaSyBU-vXpPlrm2_UBJYLzHznpvc_hhYORT8I')

gmap3.scatter(latitude_list, longitude_list, '#FF0000', size=100, marker=True)

for i in range(8):  # n^2
    for j in range(8):
        if i == j:
            continue
        else:
            lat = [latitude_list[i], latitude_list[j]]
            lon = [longitude_list[i], longitude_list[j]]
            gmap3.plot(lat, lon, 'red', edge_width=2)

gmap3.draw(
    "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/html/map_before.html")

###########################################AFTER####################################################

latitude_list = [2.7456, -6.1275, 22.3080, 25.0797,
                 35.5494, 37.4602, 39.5098, 13.6900, 2.7456]
longitude_list = [101.7072, 106.6537, 113.9185, 121.2342,
                  139.7798, 126.4407, 116.4105, 100.7501, 101.7072]

gmap2 = gmplot.GoogleMapPlotter(
    2.7456, 101.7072, 4, apikey='AIzaSyBU-vXpPlrm2_UBJYLzHznpvc_hhYORT8I')

gmap2.scatter(latitude_list, longitude_list, '#FF0000', size=100, marker=True)
gmap2.plot(latitude_list, longitude_list, 'red', edge_width=2.5)

gmap3.draw(
    "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/html/map_after.html")
