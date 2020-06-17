# Project Description 2
Algorithm Design and Analysis (WIA2003) Assignment, 2019/2020 Session

One of the essences of computer science and information technology is to solve problem faced by human-kind. As the outcome of this project, you are required develop a computer program that is able to resolve the following problems:

## Problem 1
Ben Sherman is a UK broker who is looking for industrial investment opportunities in the cities of Asia. He already invested in a company in Kuala Lumpur and now he plans to travel to several cities in Asia from Kuala Lumpur to expand his investment. The cities include Jakarta, Bangkok, Taipei, Hong Kong, Tokyo, Beijing and Seoul. 

1.	Get and mark locations of the all the cities Ben plan to visit.
    
    a.	Guide 1: you can use Python Geocoding Toolbox
    Look up: https://pypi.python.org/pypi/geopy#downloads
    
    b.	Guide 2: you can use gmplot
    Look up: https://github.com/vgm64/gmplot 

2.	Get the distances between each of these destinations

    a.	Guide 1: you can use Python Geocoding Toolbox
    
    b.	Suggestion 2: you should use Google Distance Matrix API
        i.	Login to the google developer’s website and follow through the examples. It is important that you know how to use 
            the API key given to you within the code that you are going to use. Refer to this link: 
            https://developers.google.com/maps/documentation/distance-matrix/start
            
3.	Journey planner: Suggest a journey for Ben to visit each of the cities once with the least distance travelled. 

4.	Plot line between the destinations.

    a.	Guide1:  you can use google.maps.Polyline. You can refer to this link:
    https://www.sitepoint.com/create-a-polyline-using-the-geolocation-and-the-google-maps-api/
    
## Problem 2
Ben decided to focus more on the possibilities of better return of investment in cities which has a positive economy and financial growth. So, Ben needs to do some analysis of local economy and finance situation for the last 3 months.

5.	Extract information from major economic/financial news website for each city. You need to find 5 related articles within the last 3 months to be analysed.  

    a.	Sometimes a webpage must be converted to the text version before it can be done
        i.	Guide 1: You may refer to this website to extract word from a website
        https://www.textise.net/ 
    b.	Guide 2: You may refer to this website on how to count word frequency in a website
    https://programminghistorian.org/lessons/counting-frequencies 
    c.	You can also filter stops words from the text you found
        i.	Guide 3: Stops words are such as conjunctions and prepositions. You may refer to this link: 
            https://www.ranks.nl/stopwords 
        ii.	Program using Rabin-karp algorithm to find and delete the stop words.
        
6.	Plot line/scatter/histogram graphs related to the word count using Plotly (Word count, stop words)
    d.	Guide 3: You may refer this link on how to install Plotly and how to use the API keys
     http://www.instructables.com/id/Plotly-with-Python/ 
    https://plot.ly/python/getting-started/ 
    
7.	Compare words in the webpages with the positive, negative and neutral English words using a String-Matching algorithm
    e.	Guide 4: Use the following word as positive and negative English words
    http://positivewordsresearch.com/list-of-positive-words/
    http://positivewordsresearch.com/list-of-negative-words/ 
    f.	Put these words in a text file for you to access them in your algorithm
    g.	Words that are not in the list can be considered as neutral
    
8.	Plot histogram graphs of positive and negative words found in the webpages.
    h.	Guide 5: Use Plotly
    
9.	Give an algorithmic conclusion regarding the sentiment of those articles
    i.	Guide 6: If there are more positive words, conclude that the article is giving positive sentiment, if there are more negative words, conclude that the article is giving negative sentiment.
    j.	You may try to conclude in different perspectives such as whether the list of positive and negative words above is accurate to be used in the context of the article you extracted the text.
    k.	Based on the conclusion, you may say the country has positive or negative economic/financial situation.

## Problem 3
Ben realised that he needs to optimise his travel. He will give priority to cities with possible better investment return based on the analysis of local economic and financial situation. If next nearest city to be visited have less better economic and financial situation than any of the other cities, Ben will visit other city first provided that the difference of distance between the 2 cities is not more than 40% and the difference of sentiment analysis between the 2 cities is not less than 2%. 

10.	Calculate the total probability distribution of possible routes. Then, write the summary of all possible route for Ben to take, ranking from the most recommended to the least recommended.
