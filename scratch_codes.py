#THIS ARE THE SCRATCH CODE AND TEST CODES


#URL and links removal
testvalue = re.sub(r'http\S+', '', df_reviews["Product_Review"])
testvalue = re.sub(r"x000D",'',testvalue)




def preprocess_cleaning(textcomment):
    #URL and links removal
    testvalue = re.sub(r'http\S+', '', df["Product_Review"][2])
    testvalue = re.sub(r"x000D",'',testvalue)

    #html tag removal
    tag_rem = re.compile(r'<[^>]+>')
    testvalue = tag_rem.sub('', testvalue)

    #punctuation and character removal
    testvalue = re.sub('[^a-zA-Z0  -9]',' ', testvalue)

    #Single Character Removal
    testvalue = re.sub(r"\s+[a-zA-Z]\s+", ' ', testvalue)

    #Multiple Spaces Removal
    testvalue = re.sub(r'\s+', ' ', testvalue)

    #Stopword Removal
    stopword_pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    testvalue = stopword_pattern.sub('', testvalue)
    print (testvalue)
    return testvalue