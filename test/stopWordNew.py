import string

# Given a list of words, remove any that are in a list of stop words.
def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

directory = "/Users/muhdrahiman/Desktop/SE 18:19/S4/[WIA2005] (ALGORITHM DESIGN AND ANALYSIS)/GROUP ASSIGNMENT/QUESTION 2/textfiles/"
# directory = "/Users/Faidz Hazirah/OneDrive/Documents/SEM 4/ALGO/Project/"
filesArr = ["jak.txt","bkk.txt","hkg.txt","tpe.txt","tok.txt","kor.txt","pek.txt"]
stopWordsDirectory = directory + "stopwords.txt"
wordCount_before = [0,0,0,0,0,0,0]
index = 0

print("################################################ STOP WORDS REMOVAL ################################################")
for i in range(0, len(filesArr), 1):
    fileDirectory = directory + filesArr[index]
    
    # Count total words after stop words removal
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

    for i in words:
        totalCount = totalCount + 1
    
    print("Total word count of {0} after stop words removal: {1}\n".format(filesArr[index], totalCount))

    index = index + 1

print("################################################ +VE/-VE WORDS ANALYSIS ################################################")