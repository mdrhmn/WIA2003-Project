# import gmplot package
import gmplot

latitude_list = [2.7456, -6.1275, 13.6900, 22.3080, 25.0797, 35.5494, 37.4602, 39.5098 ] 
longitude_list = [101.7072, 106.6537, 100.7501, 113.9185, 121.2342, 139.7798, 126.4407, 116.4105 ] 
  

gmap3 = gmplot.GoogleMapPlotter(2.7456, 101.7072, 13) 
  
# scatter method of map object  
# scatter points on the google map and display the marker
gmap3.scatter( latitude_list, longitude_list, '#FF0000', 
                              size = 100, marker = True ) 

gmap3.draw("/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/main/map_markers.html")
