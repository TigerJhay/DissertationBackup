from flask import Flask, session, render_template, request, flash, jsonify, url_for
import google.generativeai as genai
from io import StringIO
import pandas as pd 
import numpy as np
from numpy import array
import nltk #from wordcloud import wordcloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from matplotlib import pyplot as plt
# from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
lemmatizer = WordNetLemmatizer()
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch
import gc
import torch

app = Flask(__name__)

gc.collect()
mysqlconn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="dbmain_dissertation"
)
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)

input_size = 1
hidden_size = 64
num_layers = 1
num_classes = 2
sequence_length = 50

@app.route("/generaterecomendation", methods=["GET", "POST"])
def modelrecommendation():
    brands = session["brands"]
    type = session["type"]
    model = session["model"]
    # model = str(request.form["gadgetModel"])
    # brands = "Google"
    # type = "Smartphone"
    # model = "Pixel 4a"

    complete_gadget = brands + " " + type + " " + model
    item_desc = brands +  " " + model
    mysqlconn.reconnect()
    sqlstring = "SELECT * FROM gadget_reviews where Brand='" +brands+"' and Type='"+type+"' and Model='"+model+"'"
    temp_df = pd.read_sql(sqlstring, mysqlconn)
    temp_df = sub_datacleaning(temp_df)

    # This will train/test all record in the database
    sqlstring_cm = "SELECT * FROM gadget_reviews"
    temp_df_cm = pd.read_sql(sqlstring_cm, mysqlconn)
    temp_df_cm = sub_datacleaning(temp_df_cm)

    attrib_table(temp_df)
    top_reco, k_count = sub_KMeans(type)
    summary_reco, featured_reco, detailed_reco = sub_recommendation_summary(model)
    attrib_graph(summary_reco)
    airesult = sub_AIresult(item_desc)
    dev_images1,dev_images2,dev_images3,dev_images4 = sub_OpenAI(model, type, brands)
    shop_loc_list = sub_AIresult_Shop_Loc(item_desc)

    # Need to load the model here
    import torch
    lstm_model = torch.load('model.pth')
    lstm_model.eval()

    # train_loss, train_accs, test_loss, test_accs = sub_LSTM(temp_df)
    # evaluate_lstm_test_train_result(train_accs, test_accs, train_loss, test_loss)
    train_loss, train_accs, test_loss, test_accs = 0,0,0,0
    return render_template("index.html",
                        shop_loc_list = shop_loc_list,
                        dev_images1 = dev_images1,
                        dev_images2 = dev_images2,
                        dev_images3 = dev_images3,
                        dev_images4 = dev_images4,
                        ai_result = airesult,
                        str_recommendation = summary_reco,
                        str_featreco = featured_reco,
                        str_details = detailed_reco,
                        complete_gadget = complete_gadget,
                        top_reco = top_reco,
                        k_count = k_count,
                        epoch_train_losses = train_loss,
                        epoch_train_accs = train_accs,
                        epoch_test_losses = test_loss,
                        epoch_test_accs = test_accs,
                        summary_graph = "./static/HTML/images/Summary_Graph.png"
                        )

@app.route("/uploadCSV", methods=["GET", "POST"])

def uploadCSV():
    filepath = request.files["csvfile"]
    csv_string = filepath.stream.read().decode("utf-8")
    #df = pd.read_csv(StringIO(csv_string))
    try: 
        for chunk in pd.read_csv(StringIO(csv_string), chunksize=2000):
            chunk.to_sql(name="gadget_reviews", con=sqlengine, if_exists="append", index=False)
    except Exception as e:
        print(e)
        # print(csv_string + "value is this")

    return render_template("newdataset.html")

@app.route("/imgURLUpload", methods=["GET", "POST"])
def addImageURL():
    if str(request.form["txturldisplay1"]) != "":
        imagepath = str(request.form["txturldisplay1"])
    else:
        imagepath= "./static/HTML/images/NIA.jpg"
    if str(request.form["txturldisplay2"]) != "":
        imagepath2 = str(request.form["txturldisplay2"]) 
    else:
        imagepath2="./static/HTML/images/NIA.jpg"
    if str(request.form["txturldisplay3"]) != "":
        imagepath3 = str(request.form["txturldisplay3"]) 
    else:
        imagepath3="./static/HTML/images/NIA.jpg"    
    if str(request.form["txturldisplay4"]) != "":
        imagepath4 = str(request.form["txturldisplay4"]) 
    else:
        imagepath4="./static/HTML/images/NIA.jpg"

    brand =  session["ndsbrands"]
    type = session["ndstype"]
    model = session["ndsmodel"]
    mysqlconn.reconnect()
    cursor = mysqlconn.cursor()
    sqlstring = "INSERT INTO image_paths (Model, Brand, Type, Path,Path2,Path3,Path4) VALUES (%s,%s,%s,%s,%s,%s,%s)"    
    print (sqlstring)
    strvalue = (model, brand, type, imagepath,imagepath2,imagepath3,imagepath4,)
    cursor.execute(sqlstring, strvalue)
    notif = "Image Uploaded and save to database"
    mysqlconn.commit()
    mysqlconn.close()
    return render_template("newdataset.html", notif=notif)

@app.route("/newdataset")
def index():
    mysqlconn.reconnect()
    temp_df = pd.read_sql("SELECT Distinct(Brand) FROM gadget_reviews" , mysqlconn)
    brands = temp_df["Brand"].drop_duplicates()
    mysqlconn.close()
    return render_template("newdataset.html",brands = brands.to_numpy())

@app.route("/ndsbrandtype", methods=["GET", "POST"])
def ndsbrandtype():
    session["ndsbrands"]= str(request.form["ndsgadgetBrand"])
    mysqlconn.reconnect()
    temp_df = pd.read_sql("SELECT Distinct(Type) FROM gadget_reviews where Brand='" +session["ndsbrands"] +"'", mysqlconn)
    gadgetType = temp_df["Type"].drop_duplicates()
    return render_template("newdataset.html", 
                           gadgetType = gadgetType.to_numpy(), 
                           selectbrand= session["ndsbrands"])
 
@app.route("/ndstypemodel", methods=["GET", "POST"])
def ndstypemodel():
    session["ndstype"]= str(request.form["gadgetType"])
    mysqlconn.reconnect()
    temp_df = pd.read_sql("SELECT Distinct(Model) FROM gadget_reviews where Brand='" +session["ndsbrands"] +"' and Type='"+session["ndstype"]+"'", mysqlconn)
    gadgetModel = temp_df["Model"].drop_duplicates()
    return render_template("newdataset.html", 
                           gadgetModel = gadgetModel.to_numpy(), 
                           selectedtype = session["ndstype"], 
                           selectbrand= session["ndsbrands"])

@app.route("/ndsmodelcomplete", methods=["GET", "POST"])
def ndsmodelcomplete():
    session["ndsmodel"] = str(request.form["gadgetModel"])
    mysqlconn.reconnect()
    return render_template("newdataset.html", 
                           selectedtype = session["ndstype"], 
                           selectbrand= session["ndsbrands"], 
                           selectedmodel = session["ndsmodel"])

# -------------------------------------------------- 
# For Index.html
# --------------------------------------------------

@app.route("/")
def home():
    mysqlconn.reconnect()
    temp_df = pd.read_sql("SELECT Distinct(Brand) FROM gadget_reviews order by Brand" , mysqlconn)
    brands = temp_df["Brand"].drop_duplicates()
    mysqlconn.close()
    return render_template("index.html", 
                           brands = brands.to_numpy(),
                           dev_images = "/static/HTML/images/NIA.jpg",
                           summary_graph = "/static/HTML/images/NIA.jpg"
                           )

@app.route("/brandtype", methods=["GET", "POST"])

def brandtype():
    session["brands"]= str(request.form["gadgetBrand"])
    mysqlconn.reconnect()
    temp_df = pd.read_sql("SELECT Distinct(Type) FROM gadget_reviews where Brand='" +session["brands"] +"' order by Type", mysqlconn)
    mysqlconn.close()
    gadgetType = temp_df["Type"].drop_duplicates()
    return render_template("index.html", gadgetType = gadgetType.to_numpy(), selectbrand= session["brands"])
 
@app.route("/typemodel", methods=["GET", "POST"])
def typemodel():
    session["type"]= str(request.form["gadgetType"])
    mysqlconn.reconnect()
    temp_df = pd.read_sql("SELECT Distinct(Model) FROM gadget_reviews where Brand='" +session["brands"] +"' and Type='"+session["type"]+"' order by Model", mysqlconn)
    mysqlconn.close()
    gadgetModel = temp_df["Model"].drop_duplicates()
    return render_template("index.html", gadgetModel = gadgetModel.to_numpy(), selectedtype = session["type"], selectbrand= session["brands"])

@app.route("/modelcomplete", methods=["GET", "POST"])
def modelcomplete():
    session["model"] = str(request.form["gadgetModel"])
    return render_template("index.html", selectedtype = session["type"], selectbrand= session["brands"], selectedmodel = session["model"])

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def sub_recommendation_summary(model):
#    model = "iPhone 13"
    mysqlconn.close()
    mysqlconn._open_connection()
    # model = "Galaxy S24+"
    #temp_df_count = pd.read_sql("SELECT count(model) as count FROM gadget_reviews where Model='"+model+"'", mysqlconn)
    #temp_df_reco = pd.read_sql("SELECT * FROM attribute_table where Model='"+model+"'", mysqlconn)
    with sqlengine.begin() as connection:
        temp_df_count = pd.read_sql_query(sqlalch.text("SELECT count(model) as count FROM gadget_reviews where Model='"+model+"'"), connection)

    with sqlengine.begin() as connection:
        temp_df_reco = pd.read_sql(sqlalch.text("SELECT * FROM attribute_table where Model='"+model+"'"), connection)
    
    batt = temp_df_reco["Batt_PR"][0]
    scr = temp_df_reco["Scr_PR"][0]
    spd = temp_df_reco["Spd_PR"][0]
    mem = temp_df_reco["Mem_PR"][0]
    aud = temp_df_reco["Aud_PR"][0]
    featured_reco = ""
    sub_featured = ""
    if batt > scr and batt > spd and batt > mem and batt > aud:
        featured_reco += "Battery is one of the best feature."
        sub_featured += "Essentially, a gadget's battery life is a key metric for buyers, shaping their perception of the device's practicality, ease of use, and overall desirability. A long-lasting battery is a key feature for many users, especially those who are always on the go or who use their devices for long periods of time. A long battery life is also a key selling point for many devices, as it can help differentiate a product from its competitors and attract more customers. In addition, a long battery life can help improve a device's overall user experience, as it can reduce the need for frequent charging and allow users to use their devices for longer periods of time without interruption."
    elif scr > batt and scr > spd and scr > mem and scr > aud:
        featured_reco += "Screen size and/or dsplay is one of the best feature"
        sub_featured +=  "Larger, high-resolution screens provide a more immersive and enjoyable experience for watching videos, playing games, and browsing photos. A larger screen also makes it easier to read text and view images, which can be especially useful for users with poor eyesight or who use their devices for extended periods of time. In addition, a high-resolution screen can display more detail and provide a sharper, clearer image, which can enhance the overall viewing experience. A high-quality screen can also help improve a device's overall user experience, as it can make text and images easier to read and provide a more vibrant and engaging display."
    elif spd > batt and spd > scr and spd > mem and spd > aud:
        featured_reco += "Speed or response is one of the best feature"
        sub_featured +=  "A fast processor can help improve a device's overall performance and responsiveness, making it more efficient and enjoyable to use. A fast processor can help reduce lag and improve the speed of tasks such as opening apps, browsing the web, and playing games. A fast processor can also help improve a device's multitasking capabilities, allowing users to run multiple apps simultaneously without experiencing slowdowns or performance issues. In addition, a fast processor can help improve a device's overall user experience, as it can make the device more responsive and enjoyable to use."
    elif mem > batt and mem > scr and mem > spd and mem > aud:
        featured_reco += "Memory is one of the best feature"
        sub_featured += "Sufficient memory, particularly RAM (Random Access Memory), allows gadgets to handle multiple tasks simultaneously without slowing down. This is essential for smooth operation when running various apps or programs. Memory is also important for storing data, such as photos, videos, and music, as well as for running the operating system and other essential software. A gadget with sufficient memory will be able to run smoothly and efficiently, providing a better user experience."
    elif aud > batt and aud > scr and aud > spd and aud > mem:
        featured_reco += "Audio is one of the best feature"
        sub_featured += "High-fidelity audio transforms the experience of watching movies, listening to music, and playing games. Clear, rich sound creates a more immersive and engaging environment. High-quality audio can also enhance the overall user experience, making it more enjoyable and satisfying. In addition, high-fidelity audio can help improve a device's overall performance, as it can provide a more realistic and engaging sound experience. High-quality audio can also help differentiate a product from its competitors and attract more customers."
    else:
        featured_reco += "Neither of the features is good or bad"
        sub_featured += "Over all, the gadget is neither good nor bad. It is just an average gadget."
        
    #summary_reco = "Based on the " + str(temp_df_count["count"][0]) + " reviews: \n Battery has " + str(temp_df_reco["Batt_PR"][0]) + " positive reviews \n Screen has " + str(temp_df_reco["Scr_PR"][0]) + " positive reviews \n Speed has " + str(temp_df_reco["Spd_PR"][0]) + " positive reviews \n Memory Size has " + str(temp_df_reco["Mem_PR"][0]) + " positive reviews \n Audio Quality " + str(temp_df_reco["Aud_PR"][0]) + " positive reviews "
    summary_reco = [ temp_df_reco["Batt_PR"][0], temp_df_reco["Scr_PR"][0], temp_df_reco["Spd_PR"][0], temp_df_reco["Mem_PR"][0], temp_df_reco["Aud_PR"][0] ] 
    #summary_reco = summary_reco.split("\n")
    return summary_reco, featured_reco, sub_featured 
    
def sub_AIresult(item_desc):
        #item_desc = "Apple iphone 15"
        genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
        model = genai.GenerativeModel("gemini-1.5-flash")
        airesult = str(model.generate_content("specifications of " + item_desc).text)
        airesult = airesult.replace("\n","")
        airesult = airesult.replace("**","<br>")
        airesult = airesult.replace("*","")
        return airesult

def sub_AIresult_Shop_Loc(item_desc):
    import google.generativeai as genai
    #item_desc = "Apple iphone 15"
    genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
    model = genai.GenerativeModel("gemini-1.5-flash")
    shoploc_list = str(model.generate_content( "list of stores to buy " + item_desc + " in the philippines").text)    
    shoploc_list = shoploc_list.split("**")
    shoploc_list = [strvalue.replace("\n","") for strvalue in shoploc_list]
    shoploc_list = [strvalue.replace("*","<br>") for strvalue in shoploc_list]
    return shoploc_list
    
def sub_OpenAI(model, type, brand):
    # brand = "Apple" 
    # type = "Smartphone"
    # model = "iPhone 15"
    cursor = mysqlconn.cursor()
    cursor.execute("SELECT Path, Path2, Path3, Path4 FROM image_paths where model='" + model + "' and type='"+type+ "' and brand='"+brand+"'")
    img_result = cursor.fetchone()
    if img_result is not None:
        fetch_img_result1 = img_result[0]
        fetch_img_result2 = img_result[1]
        fetch_img_result3 = img_result[2]
        fetch_img_result4 = img_result[3]
    else:
        fetch_img_result1 = "./static/HTML/images/NIA.jpg"
        fetch_img_result2 = "./static/HTML/images/NIA.jpg"
        fetch_img_result3 = "./static/HTML/images/NIA.jpg"
        fetch_img_result4 = "./static/HTML/images/NIA.jpg"
    cursor.close()
    return fetch_img_result1, fetch_img_result2, fetch_img_result3, fetch_img_result4
        
def sub_datacleaning(temp_df):
        # custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

        #Remove Column Username since this column is unnecessary
        temp_df["Reviews"] = temp_df["Reviews"].str.lower()
        
        # Checking for missing values. Fill necessary and drop if reviews are null
        if temp_df["Username"].isnull().values.any() == True:
            temp_df["Username"] = temp_df["Username"].fillna("No Username")       
        
        # Date with invalid values will be default to 1/1/11, which also not useful :)
        # Date with no values will also be converted, which also not useful :)
        if temp_df["Date"].isnull().values.any() == True:
            temp_df["Date"] = temp_df["Date"].fillna("1/1/11")

        # All records with not value for REVIEWS will be dropped
        if temp_df["Reviews"].isnull().values.any() == True:
            temp_df = temp_df.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

        # Replace all special characters into black spaces which will also be remove
        temp_df["Reviews"] = temp_df["Reviews"].str.replace("\n",' ')
        temp_df["Reviews"] = temp_df["Reviews"].str.replace("\r",' ')
        temp_df["Reviews"] = temp_df["Reviews"].replace(r'http\S+', '', regex=True)
        temp_df["Reviews"] = temp_df["Reviews"].replace(r"x000D", '', regex=True)
        temp_df["Reviews"] = temp_df["Reviews"].replace(r'<[^>]+>', '', regex= True)        
        temp_df["Reviews"] = temp_df["Reviews"].replace('[^a-zA-Z0-9]', ' ', regex=True)
        temp_df["Reviews"] = temp_df["Reviews"].replace(r"\s+[a-zA-Z]\s+", ' ', regex=True) #Eto
        temp_df["Reviews"] = temp_df["Reviews"].replace(r" +", ' ', regex=True)

        # temp_df_temp = temp_df
        # # Eto nalang
        # temp_df = temp_df_temp

        def tokenize_reviews(review_text):
            review_sentence = word_tokenize(review_text)
            return review_sentence
        temp_df['Reviews'] = temp_df['Reviews'].apply(tokenize_reviews)


        # nltk.download('stopwords')
        # def remove_stopwords(review_text):                  
        #     stop_words = set(stopwords.words('english'))
        #     filtered_text = [word for word in review_text if word not in stop_words]
        #     return filtered_text
        # temp_df['Reviews'] = temp_df['Reviews'].apply(remove_stopwords)

        nltk.download('wordnet')
        nltk.download('punkt_tab')
        nltk.download('omw-1.4')
        def lemmatize_review(review_text):
            lemmatizer = WordNetLemmatizer()
            lemmatize_words = [lemmatizer.lemmatize(word) for word in review_text]
            lemmatize_text = ' '.join(lemmatize_words)
            return lemmatize_text
        temp_df['Reviews'] = temp_df['Reviews'].apply(lemmatize_review)
        
        temp_df["Reviews"].replace('', None, inplace=True)
        
        if temp_df["Reviews"].isnull().values.any():
            temp_df = temp_df.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

        #Rating of the sentiments will be converted into 3 classes
        # 0 - Negative Rating or review, These are with rating of 1 & 2
        # 1 - Positive Rating or review, These are with rating of 4 & 5
        
        temp_df["Rating"] = temp_df["Rating"].astype(str)
        temp_df["Rating"] = temp_df["Rating"].str.replace('[1-2]', '0', regex=True)
        temp_df["Rating"] = temp_df["Rating"].str.replace('[4-5]', '1', regex=True)
        temp_df["Rating"] = temp_df["Rating"].astype(int)
        # 3 - Neutral Rating or review, These are with rating of 3
        # This rating will be drop to be dataframe since these are all neither positive or negative
        temp_df = temp_df.drop(temp_df[temp_df["Rating"]==3].index, inplace=False)
        return temp_df

def attrib_table(temp_df_attrib):
    
    #--------------------------------------------------------------------------------------------
    #Extracting phrases for creating corpora that will be use in decision tree recommendation
    # FF: temp_df_attrib here is a cleaned dataset came from datacleaning function
    #--------------------------------------------------------------------------------------------
    #temp_df_attrib = temp_df
    df_reviews = temp_df_attrib.drop(axis=1, columns=["Date"])
    df = pd.DataFrame()
    def extract_attrib(attrib_value):
        df_temp = df_reviews.loc[df_reviews["Reviews"].str.contains(attrib_value, regex=False)]
        df_temp["Reviews"] = df_temp["Reviews"].str.replace('[0-9]', "", regex=True)
        
        if attrib_value == "battery":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}battery\b(?:\W+\w+){0,2})')
        elif attrib_value == "speed":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}speed\b(?:\W+\w+){0,2})')
        elif attrib_value == "memory":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}memory\b(?:\W+\w+){0,2})')
        elif attrib_value == "screen":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}screen\b(?:\W+\w+){0,2})')
        elif attrib_value == "screen":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}audio\b(?:\W+\w+){0,2})')
        else:
            df_temp["Attribute"] = attrib_value
        
        df_temp = df_temp.dropna(axis=0, subset=["Reviews"], how='any')
        df_temp = df_temp.drop_duplicates(subset="Reviews")
        df_temp["Attribute"] = attrib_value
        return df_temp

    list_attrib = ["battery", "screen", "speed", "memory", "audio"]
    for attrib in list_attrib:
        df = pd.concat([df, extract_attrib(attrib)])

    attrib_matrix = pd.DataFrame(columns=["Model", "Batt_PR","Batt_NR", "Scr_PR", "Scr_NR", "Spd_PR", "Spd_NR", "Mem_PR", "Mem_NR", "Aud_PR", "Aud_NR"])
    gadget_list = df_reviews["Model"].unique()

    def convert_to_matrix(gadget_model):
        df_model = df.loc[df["Model"].str.contains(gadget_model)]
        df_rev = df_model.loc[df_model["Reviews"].str.contains("battery")]
        batt_rpos = df_rev["Rating"].value_counts().get(1,0)
        batt_rneg = df_rev["Rating"].value_counts().get(0,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("screen")]
        scr_rpos = df_rev["Rating"].value_counts().get(1,0)
        scr_rneg = df_rev["Rating"].value_counts().get(0,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("speed")]
        spd_rpos = df_rev["Rating"].value_counts().get(1,0)
        spd_rneg = df_rev["Rating"].value_counts().get(0,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("memory")]
        mem_rpos = df_rev["Rating"].value_counts().get(1,0)
        mem_rneg = df_rev["Rating"].value_counts().get(0,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("audio")]
        aud_rpos = df_rev["Rating"].value_counts().get(1,0)
        aud_rneg = df_rev["Rating"].value_counts().get(0,0)

        row_value = [gadget_model, batt_rpos, batt_rneg, scr_rpos, scr_rneg, spd_rpos, spd_rneg, mem_rpos, mem_rneg, aud_rpos, aud_rneg]    
        return row_value

    for colname in gadget_list:
        attrib_matrix.loc[len(attrib_matrix)] = convert_to_matrix(colname)
    attrib_matrix.to_sql(con=sqlengine, name="attribute_table", if_exists='replace', index=True)

def attrib_graph(data_count):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    #data = [100, 27, 20, 1,1]
    gadgetnames = ['Battery', 'Screen', 'Speed', 'RAM', 'Audio']
    bar_labels = ['Battery', 'Screen', 'Speed', 'RAM','Audio']
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

    ax.bar(gadgetnames, data_count, label=bar_labels, color=bar_colors)
    ax.set_ylabel('User Reviews')
    ax.set_title('Summary of User Gadget Reviews')
    ax.legend(title='Gadget Labels')
    plt.savefig(".\static\HTML\images\Summary_Graph.png")
    plt.close()
            
def sub_LSTM(temp_df):   
#---------------------------------------
# This portion is for LSTM algorithm
#---------------------------------------
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from gensim.models import Word2Vec
    
    embedding_size = 50
    SEQUENCE_LENGTH = 50
    batch_size = 50
    epochs = 20

    #Tokenize all words in the dataframe
    temp_df["Reviews"] = temp_df["Reviews"].apply(word_tokenize)

    df_train, df_test = train_test_split(temp_df, test_size=.2, random_state=42)
    # df_train['Rating'].value_counts()
    # df_test['Rating'].value_counts()

    all_reviews = df_train['Reviews'].tolist()
    all_reviews.extend(df_test['Reviews'].tolist())

    #Train Word2Vec
    wordvector_model = Word2Vec(all_reviews, vector_size = embedding_size)
    
    # Word2index dictionary
    word2idx = {word: i+1 for i, word in enumerate(wordvector_model.wv.index_to_key)}
    #word2idx["<PAD>"] = 0

    #wv['_____'] the value inside wv is the value needed for prediction
    wordvector_model.wv[word2idx]
    # value = wordvector_model.wv.most_similar('bit', topn=3)

    #Embedding Matrix
    def convert_sequences_to_tensor(sequences, num_tokens_in_sequence, embedding_size):
        num_sequences = len(sequences)
        print((num_sequences, num_tokens_in_sequence, embedding_size))
        
        data_tensor = torch.zeros((num_sequences, num_tokens_in_sequence, embedding_size))
        
        for index, review in enumerate(list(sequences)):
            # Create a word embedding for each word in the review (where a review is a sequence)
            truncated_clean_review = review[:num_tokens_in_sequence] # truncate to sequence length limit
            list_of_word_embeddings = [wordvector_model.wv[word] if word in wordvector_model.wv else [0.0]*embedding_size for word in truncated_clean_review]

            # convert the review to a tensor
            sequence_tensor = torch.FloatTensor(list_of_word_embeddings)

            # add the review to our tensor of data
            review_length = sequence_tensor.shape[0] # (review_length, embedding_size)
            data_tensor[index,:review_length,:] = sequence_tensor
        
        return data_tensor

    train_data_X = convert_sequences_to_tensor(df_train['Reviews'].to_numpy(), SEQUENCE_LENGTH, embedding_size)
    train_data_y = torch.FloatTensor([int(d) for d in df_train['Rating'].to_numpy()])
    test_data_X = convert_sequences_to_tensor(df_test['Reviews'].to_numpy(), SEQUENCE_LENGTH, embedding_size)
    test_data_y = torch.FloatTensor([int(d) for d in df_test['Rating'].to_numpy()])

    train_data = TensorDataset(train_data_X, train_data_y)
    test_data = TensorDataset(test_data_X, test_data_y)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    #Need to use the resources of CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device
        
    import torch.nn as nn
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers, drop_prob=0.7):
            super(LSTMModel,self).__init__()

            self.num_stacked_layers = num_stacked_layers
            self.hidden_size = hidden_size

            self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = num_stacked_layers,
                batch_first = True)
            
            self.dropout = nn.Dropout(drop_prob) # randomly sets outputs of a tensor to 0 during training
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            batch_size = x.size(0)
            
            # Initialize the cell state and hidden state
            h0 = torch.zeros((self.num_stacked_layers, batch_size, self.hidden_size)).to(device)
            c0 = torch.zeros((self.num_stacked_layers, batch_size, self.hidden_size)).to(device)

            # Call the LSTM
            lstm_out, hidden = self.lstm(x, (h0, c0))

            # contiguous() moves all data into 1 block of memory on the GPU
            # (batch_size, sequence_size, embedding_size) -> (batch_size*sequence_size, embedding_size)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

            # dropout and fully connected layer
            lstm_out = self.dropout(lstm_out) # Only during training
            fc_out = self.fc(lstm_out)
            
            # apply the sigmoid function to maps the value to somewhere between 0 and 1
            sigmoid_out = self.sigmoid(fc_out)

            # reshape to be batch_size first - every batch has a value between 0 and 1
            sigmoid_out = sigmoid_out.view(batch_size, -1) # a list of lists with single elements
            sigmoid_out = sigmoid_out[:, -1] # get the output labels as a list

            # return last sigmoid output and hidden state
            return sigmoid_out, hidden 

    LSTM_INPUT_SIZE = embedding_size # size of the embeddings
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_STACKED_LAYERS = 2

    lstm_model = LSTMModel(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_STACKED_LAYERS)
    lstm_model.to(device)
    # print(lstm_model)

    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    
    def accuracy(pred, label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    def train_loop(model, train_loader, optimizer, criterion):
        model.train()
        train_accuracy = 0.0
        train_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, h = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate the loss
            optimizer.zero_grad() # Clear out all previous gradients
            loss.backward() # Calculate new gradients
            optimizer.step() # Update parametres using the gradients

            train_losses.append(loss.item())
            train_accuracy += accuracy(outputs, labels)

            epoch_train_loss = np.mean(train_losses) * 100.00
            epoch_train_acc = (train_accuracy/len(train_loader.dataset))*100.0
        return (epoch_train_loss, epoch_train_acc)

    # Test/Validation Loop
    def test_loop(model, test_loader, criterion):
        model.eval()
        test_accuracy = 0.0
        test_losses = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, val_h = model(inputs)
                loss = criterion(outputs, labels)
                test_losses.append(loss.item())
                test_accuracy += accuracy(outputs, labels)

        epoch_test_loss = np.mean(test_losses) * 100.0
        epoch_test_accuracy = (test_accuracy/len(test_loader.dataset))*100.0

        return (epoch_test_loss, epoch_test_accuracy)

    # Training and validation loop
    epoch_train_losses = []
    epoch_train_accs = []
    epoch_test_losses = []
    epoch_test_accs = []

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = train_loop(lstm_model, train_loader, optimizer, criterion)
        train_loss = round(epoch_train_loss,3)
        train_accs = round(epoch_train_acc,3)

        epoch_test_loss, epoch_test_acc = test_loop(lstm_model, test_loader, criterion)
        test_loss = round(epoch_test_loss,3)
        test_accs = round(epoch_test_acc,3)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.4f} | Test Loss: {epoch_test_loss:.4f} Test Acc: {epoch_test_acc:.4f}')
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(epoch_train_acc)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_accs.append(epoch_test_acc)

    
    
    # Model Integration
    # sub_LSTM_pickle(lstm_model)

    # # Pytorch model
    torch.save(lstm_model.state_dict(), 'model.pth')
    # lstm_model = LSTMModel()
    # lstm_model.load_state_dict(torch.load('model.pth'))




        # TESTING AND TRAINING GRAPH
    evaluate_lstm_test_train_result(epoch_train_accs,epoch_test_accs,epoch_train_losses, epoch_test_losses)
    
    #CONFUSION MATRIX DISPLAY
    evaluate_lstm_model_pytorch(lstm_model,test_loader,label_encoder=None, device='cpu')

    return train_loss, train_accs, test_loss, test_accs

def sub_LSTM_pickle(model_name):
    print =("Testing Pickel Function")
    model = model_name
    import joblib
    joblib.dump(model, "Actual_LSTM_model.joblib")
#--------------------------------------------------------------------
# CONFUSION MATRIX, PRECISION, RECALL AND F1 SCORE
#--------------------------------------------------------------------
def evaluate_lstm_test_train_result(epoch_train_accs, epoch_test_accs, epoch_train_losses, epoch_test_losses):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (10, 3))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_train_accs, label='Train Accuracy')
    plt.plot(epoch_test_accs, label='Test Accuracy')
    plt.title("Train and Test Accuracy of LSTM model")
    plt.ylabel("Accuracy Percentage")
    plt.xlabel("No. of Epochs")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_train_losses, label='Train Loss')
    plt.savefig("static\HTML\images\LSTM_train_acc.png")
    plt.plot(epoch_test_losses, label='Test Loss')
    plt.savefig("static\HTML\images\LSTM_test_acc.png")
    plt.title("Train and Test Losses of LSTM Model")
    plt.ylabel("Loss Percentage")
    plt.xlabel("No. of Epochs")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_lstm_model_pytorch(lstm_model, test_loader, label_encoder, device='cpu'):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    lstm_model.to(device)
    lstm_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = lstm_model(inputs)
            predicted = (outputs > 0.5).long()  # Convert sigmoid output to binary
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Precision, Recall, F1 Score
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    #Convert Values into percent
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues")   
    ax.set_title("Confusion Matrix for LSTM Model")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix")
    plt.xticks(np.arange(2)+0.5,["Not Recommended", "Recommended"])
    plt.yticks(np.arange(2)+0.5,["Not Recommended", "Recommended"])
    plt.show()

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

def sub_KMeans(gadgettype):
    mysqlconn.reconnect()
    kmeans_df = pd.read_sql("SELECT * FROM gadget_reviews where Type='" + gadgettype + "'", mysqlconn)
    kmeans_df = kmeans_df.iloc[:10000,:]
    df_reco = kmeans_df[["Rev_No",'Model', 'Rating']]

    pivot_table = pd.pivot_table(df_reco, index='Rev_No', columns="Model", values='Rating', fill_value=0)
    num_clusters = 3  # Choose the number of clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pivot_table)
    user_id = 12
    user_cluster_label = cluster_labels[user_id - 1]
    users_in_same_cluster = pivot_table.index[cluster_labels == user_cluster_label]
    average_ratings = pivot_table.loc[users_in_same_cluster].mean()
    sorted_ratings = average_ratings.sort_values(ascending=False)
    k = 4
    top_kmeans_reco = sorted_ratings.head(k)

    return top_kmeans_reco.items(), k

def sub_decision_tree(gadgettype):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt

    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    mysqlconn.reconnect()
    # dectree_df = pd.read_sql("SELECT * FROM gadget_reviews WHERE Type = '" + gadgettype + "'", mysqlconn)
    dectree_df = pd.read_sql("SELECT * FROM gadget_reviews", mysqlconn)
    dectree_df = sub_datacleaning(dectree_df)
    X  = dectree_df["Reviews"]

    countvector = CountVectorizer()
    X = countvector.fit_transform(dectree_df["Reviews"])
    y = dectree_df["Rating"].astype("float16")

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=.2)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    y_pred = dtc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    #Convert Values into percent
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues")    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix for Decision Tree (%)")
    plt.xticks(np.arange(2)+0.5,["Not Recommended", "Recommended"])
    plt.yticks(np.arange(2)+0.5,["Not Recommended", "Recommended"])
    plt.show()
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["Not Recommend", "Recommend"])
    # disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix Decision Tree")
    # plt.show()

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.3%}")
    print(f"Recall: {recall:.3%}")
    print(f"F1 Score: {f1:.3%}")

app.secret_key = "abcdef12345"
if __name__ == "__main__":
    app.run(debug=True)
