# ----------------------------------------
# THIS ARE THE SCRATCH CODE AND TEST CODES
# ----------------------------------------


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







#vectorize = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopwords.words('english'))

vectorize = CountVectorizer()
x_val = vectorize.fit_transform(df_reviews['Reviews'])
vectorize.get_feature_names_out()
x_val
y_val = df_reviews['Rating']
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=0)
x_train_count = vectorize.fit_transform(x_train)
classifier = naive_bayes.MultinomialNB()
classifier.fit(x_train, y_train)