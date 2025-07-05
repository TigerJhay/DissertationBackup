from flask import Flask, session, render_template, request, redirect, url_for
import google.generativeai as genai
from io import StringIO
import pandas as pd 
import numpy as np
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
lemmatizer = WordNetLemmatizer()
import gc
import torch
import torch.nn as nn
import joblib 
import json
from functools import lru_cache
import sqlalchemy as sqlalch
from sqlalchemy import text, create_engine
import  mysql.connector
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4') 
import os
gc.collect()

app = Flask(__name__)
app.secret_key = "abcdef12345"

#Initialize SQL constructor
mysqlconn = mysql.connector.connect(host="localhost", user="root", password="", database="dbmain_dissertation")
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)

#Decision Tree Algo Loading and getting values
dtc = joblib.load("DT_Model2.pkl")
countvector = joblib.load("DT_Vector2.pkl")

#Load Vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Global variables
global_table_name = "Generic"
list_attrib = []

# Repository folder for uploads
UPLOAD_FOLDER = 'static/gadgetimageuploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif', 'bmp'}
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
from werkzeug.utils import secure_filename
import uuid

#Load LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        out = torch.sigmoid(self.fc(h_n)).view(-1)
        return out
    
lstm_model = LSTMClassifier(vocab_size=len(vocab), embed_dim=32, hidden_dim=64)
lstm_model.load_state_dict(torch.load("lstm_model3.pt", map_location=torch.device("cpu")))
lstm_model.eval()


# Other Utilities
MAX_LEN = 30

def encode(text):
    tokens = word_tokenize(text.lower())
    ids = [vocab.get(t, 0) for t in tokens][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

@lru_cache(maxsize=128)
def dt_predict_recommendation(model_name):
    try:
        # model_name ="iphone 13"    
        mysqlconn.reconnect()
        model_reviews_df = pd.read_sql(f"SELECT Reviews FROM gadget_reviews WHERE Model = '{model_name}'", mysqlconn)
        if model_reviews_df.empty:
            return f"No reviews found for model: {model_name}"
        model_reviews = model_reviews_df["Reviews"] .tolist()
        model_reviews = [review.lower() if isinstance(review, str) else "" for review in model_reviews]
        model_reviews_vectorized = countvector.transform(model_reviews)
        predicted_ratings = dtc.predict(model_reviews_vectorized)
        average_rating = predicted_ratings.mean()

        if average_rating >=0:
            return average_rating
        else:
            return 0
        # if average_rating >= .5:
        #     return f"The model '{model_name}' is likely to be recommended (average predicted rating: {average_rating:.2f})."
        # else:
        #     return f"The model '{model_name}' is not likely to be recommended (average predicted rating: {average_rating:.2f})."

    except Exception as e:
        return f"An error occurred: {e}"

# @lru_cache(maxsize=128)
def lstm_predict_recommendation(gadgetmodel):

    query = f"SELECT Reviews, Rating, Model FROM gadget_reviews WHERE Model = '{gadgetmodel}'"
    mysqlconn.reconnect()
    df = pd.read_sql(query, mysqlconn)
    df = sub_datacleaning_reco(df)
                               
    if df.empty:
        return f"No data found for model: {gadgetmodel}"

    predictions = []
    with torch.no_grad():
        for _, row in df.iterrows():
            input_tensor = torch.tensor([encode(row['Model'] + " " + row['Reviews'])], dtype=torch.long)
            output = lstm_model(input_tensor)
            predictions.append(output.item())

    avg_pred = np.mean(predictions)
    return avg_pred
    # scaled_rating = round(avg_pred * 5 * 2) / 2
    # full_stars = int(scaled_rating)
    # half_star = 1 if scaled_rating - full_stars == 0.5 else 0
    # empty_stars = 5 - full_stars - half_star
    # visual_stars = '★' * full_stars + '½' * half_star + '☆' * empty_stars

    # # return f"Prediction for '{gadgetmodel}' with average prediction of {avg_pred} : {'Recommend' if avg_pred >= 0.5 else 'Not Recommend'}"
    # return (f"The gadget '{gadgetmodel}' is {'Recommend' if avg_pred >= 0.5 else 'Not Recommend'}\n"
    # f"With Rating of: {scaled_rating} / 5  {visual_stars}")

def sub_datacleaning_reco(temp_df):
    temp_df = temp_df.dropna(subset=['Reviews'])
    temp_df = temp_df[temp_df['Reviews'].str.strip() != '']
    # Replace all special characters into black spaces which will also be remove
    temp_df["Reviews"] = temp_df["Reviews"].str.replace("\n",' ')
    temp_df["Reviews"] = temp_df["Reviews"].str.replace("\r",' ')
    temp_df["Reviews"] = temp_df["Reviews"].replace(r'http\S+', '', regex=True)
    temp_df["Reviews"] = temp_df["Reviews"].replace(r"x000D", '', regex=True)
    temp_df["Reviews"] = temp_df["Reviews"].replace(r'<[^>]+>', '', regex= True)
    temp_df["Reviews"] = temp_df["Reviews"].replace('[^a-zA-Z0-9]', ' ', regex=True)
    temp_df["Reviews"] = temp_df["Reviews"].replace(r"\s+[a-zA-Z]\s+", ' ', regex=True) #Eto
    temp_df["Reviews"] = temp_df["Reviews"].replace(r" +", ' ', regex=True)
    temp_df = temp_df[temp_df["Rating"] != 3]

    return temp_df

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/imgUpload", methods=['GET', 'POST'])
def imageupload():
    print(">>> PASSED IMAGE UPLOAD AREA 1")
    uploaded_files = []
    if request.method == 'POST':
        print(">>> PASSED IMAGE UPLOAD AREA 2")
        for file_key in ['imagefileurl1', 'imagefileurl2', 'imagefileurl3', 'imagefileurl4']:
            if file_key in request.files:
                print(">>> PASSED IMAGE UPLOAD AREA 3")
                file = request.files[file_key]
                if file and allowed_file(file.filename):
                    print(">>> PASSED IMAGE UPLOAD AREA 4")
                    # Get the file extension
                    ext = file.filename.rsplit('.', 1)[1].lower()
                    # Generate a unique filename using UUID
                    unique_filename = f"{uuid.uuid4().hex}.{ext}"
                    print(unique_filename + " >> unique file name")
                    # filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
                    # file.save(file_path)
                    file_url = url_for('static', filename=f'gadgetimageuploads/{unique_filename}')
                    uploaded_files.append(file_url)

                    print("FILE KEY VALUE >> " + file_key)
                    if file_key == "imagefileurl1":
                        str_file_url1 = file_url
                    elif file_key == "imagefileurl2":
                        str_file_url2 = file_url
                    elif file_key == "imagefileurl3":
                        str_file_url3 = file_url
                    elif file_key == "imagefileurl4":
                        str_file_url4 = file_url
                    else:
                        file_url == "" 
                
                else:
                    return "Invalid file type or file too large", 400
        print(f"values >> { str_file_url1} >> { str_file_url2} >> { str_file_url3} >> { str_file_url4} ")
        brand =  session["ndsbrands"]
        gadgettype = session["ndstype"]
        model = session["ndsmodel"]

        with sqlengine.begin() as conn:
            sqlstring = text("INSERT INTO image_paths (Model, Brand, Type, Path, Path2, Path3, Path4) VALUES (:model, :brand, :type, :path1, :path2, :path3, :path4)")
            conn.execute(sqlstring, {
                'model': model,
                'brand': brand,
                'type': gadgettype,
                'path1': str_file_url1,
                'path2': str_file_url2,
                'path3': str_file_url3,
                'path4': str_file_url4
            })

    return render_template('newdataset.html', uploaded_files=uploaded_files)

@app.route("/generaterecomendation", methods=["GET", "POST"])
def modelrecommendation():
    flag = request.form['flag']

    if flag == "HTTP":
        brand = session["brands"]
        type = session["type"]
        gadgetmodel = session["model"]
    elif flag == "Alternative":
        gadgetmodel = str(request.form["otherModel"])
        print (gadgetmodel)
        sqlstring = f"SELECT Brand, Type, Model FROM gadget_reviews WHERE Model = '{gadgetmodel}'"
        print (sqlstring)
        with sqlengine.begin() as sqlconnection:
            temp_result = pd.read_sql(sqlalch.text(sqlstring), sqlconnection)
            print (temp_result)
            brand = temp_result['Brand'].iloc[0]
            type = temp_result['Type'].iloc[0]
            gadgetmodel = temp_result['Model'].iloc[0] 
        complete_gadget = brand + " " + type + " " + gadgetmodel
        print (complete_gadget + " FOR ALTERNATIVE REQUEST")

    complete_gadget = brand + " " + type + " " + gadgetmodel
    print (complete_gadget)

    item_desc = brand +  " " + gadgetmodel
    mysqlconn.reconnect()
    sqlstring = "SELECT * FROM gadget_reviews where Brand='" +brand+"' and Type='"+type+"' and Model='"+gadgetmodel + "'"
    # sqlstring = "SELECT * FROM gadget_reviews where Brand='Apple' and Type='Ear Buds' and Model='Airpods'"

    temp_df = pd.read_sql(sqlstring, mysqlconn)
    
    temp_df = sub_datacleaning(temp_df)
    print ("----->>> COMPLETED - DATA CLEANED")
    
    attrib_table(temp_df, type)
    print ("----->>> COMPLETED - ATTRIB TABLE")
    
    summary_reco, featured_reco, detailed_reco = sub_recommendation_summary(gadgetmodel)
    print ("----->>> COMPLETED - SUMMARY RECOMMENDATION")
    
    attrib_graph(summary_reco)
    print ("----->>> COMPLETED - SUMMARY RECOMMENDATION GRAPH")

    airesult = sub_AIresult(item_desc)
    print ("----->>> COMPLETED - GENERATE AI GADGET SPECS SUMMARY")

    shop_loc_list = sub_AIresult_Shop_Loc(item_desc)
    print ("----->>> COMPLETED - GENERATE AI SHOPS LOCATIONS")


    dev_images1,dev_images2,dev_images3,dev_images4 = sub_OpenAI(gadgetmodel, type, brand)
    print ("----->>> COMPLETED - IMAGES LOADED")

    top_reco, k_count = sub_KMeans(type)
    print ("----->>> COMPLETED - TOP AND K")

    dt_rating = dt_predict_recommendation(gadgetmodel)
    print ("----->>> COMPLETED - DECISION TREE RECOMMENDATION")

    lstm_rating = lstm_predict_recommendation(gadgetmodel)
    print ("----->>> COMPLETED - LSTM RECOMMENDATION/PREDICTION")

    str_rating_result = sub_generate_rating(dt_rating, lstm_rating, gadgetmodel)

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
                        str_rating_result = str_rating_result,
                        summary_graph = "./static/HTML/images/Summary_Graph.png"
                        # str_result_reco = str_result_reco,
                        # str_result_dt = str_result_dt,
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

    with sqlengine.begin() as conn:
        sqlstring = text("INSERT INTO image_paths (Model, Brand, Type, Path, Path2, Path3, Path4) VALUES (:model, :brand, :type, :path1, :path2, :path3, :path4)")
        conn.execute(sqlstring, {
            'model': model,
            'brand': brand,
            'type': type,
            'path1': imagepath,
            'path2': imagepath2,
            'path3': imagepath3,
            'path4': imagepath4
        })

    return render_template("newdataset.html", notif="List of images has been save")

@app.route("/newdataset")
def index():
    temp_fb = pd.read_sql("SELECT FBdatetime, FBComment, FBRating FROM user_feedback", sqlengine)
    fbrecord = temp_fb.to_dict(orient='records')

    avg_fbrating = pd.read_sql("SELECT avg(FBRating) FROM user_feedback", sqlengine)
    avg_fbrating = round (avg_fbrating.iloc[0,0], 2)
    temp_df = pd.read_sql("SELECT Distinct(Brand) FROM gadget_reviews" , sqlengine)
    brands = temp_df["Brand"].drop_duplicates()
    return render_template("newdataset.html",brands = brands.to_numpy(), fbrecord = fbrecord, avg_fbrating = avg_fbrating)

@app.route("/userfeedbacks")
def loaduserfeedback():
    fbrecords = pd.read_sql("SELECT FBdatetime, FBComment, FBRating FROM user_feedback", sqlengine)
    return render_template("newdataset.html", fbrecord = fbrecords.to_numpy())

@app.route("/userinput")
def userinput():
    temp_df = pd.read_sql("SELECT Distinct(Brand) FROM gadget_reviews" , sqlengine)
    brands = temp_df["Brand"].drop_duplicates()
    return render_template("UserInput.html", brands = brands.to_numpy())

@app.route("/ucbrandtype", methods=["GET", "POST"])
def ucbrandtype():
    session["ucbrands"]= str(request.form["ucgadgetBrand"])
    temp_df = pd.read_sql("SELECT Distinct(Type) FROM gadget_reviews where Brand='" +session["ucbrands"] +"'", sqlengine)
    gadgetType = temp_df["Type"].drop_duplicates()
    return render_template("UserInput.html", 
                           gadgetType = gadgetType.to_numpy(), 
                           selectbrand= session["ucbrands"])
 
@app.route("/uctypemodel", methods=["GET", "POST"])
def uctypemodel():
    session["uctype"]= str(request.form["gadgetType"])
    temp_df = pd.read_sql("SELECT Distinct(Model) FROM gadget_reviews where Brand='" +session["ndsbrands"] +"' and Type='"+session["ndstype"]+"'", sqlengine)
    gadgetModel = temp_df["Model"].drop_duplicates()
    return render_template("UserInput.html", 
                           gadgetModel = gadgetModel.to_numpy(), 
                           selectedtype = session["uctype"], 
                           selectbrand= session["ucbrands"])

@app.route("/ucmodelcomplete", methods=["GET", "POST"])
def ucmodelcomplete():
    session["ucmodel"] = str(request.form["gadgetModel"])
    return render_template("UserInput.html", 
                           selectedtype = session["uctype"], 
                           selectbrand= session["ucbrands"], 
                           selectedmodel = session["ucmodel"])

@app.route("/saveUserComment", methods=["GET", "POST"])
def ucInputSave():
    ucReview = str(request.form["ucComment"])
    ucRating = str(request.form["ucRating"])
    brand =  session["ucbrands"]
    type = session["uctype"]
    model = session["ucmodel"]

    with sqlengine.begin() as conn:
        sqlstring = text("INSERT INTO gadget_reviews (Reviews, Rating, Model, Type, Brand) VALUES (:reviews, :rating, :model, :type, :brand)")
        conn.execute(sqlstring, {
            'reviews' : ucReview,
            'rating' : ucRating,
            'model': model,
            'brand': brand,
            'type': type
        })

    return render_template("UserInput.html", notif="Review and Rating Saved")

@app.route("/saveUserFeedback", methods=["GET", "POST"])
def userfeedback():
    FBdatetime = str(request.form['FBdatetime'])
    usrFBComment = str(request.form["usrFBComment"])
    usrFBRating = str(request.form["usrFBRating"])
    brand =  session["brands"]
    type = session["type"]
    model = session["model"]

    with sqlengine.begin() as conn:
        sqlstring = text("INSERT INTO user_feedback (FBComment, FBRating, FBdatetime, Model, Type, Brand) VALUES (:FBComment, :FBRating,:FBdatetime,  :model, :type, :brand)")
        conn.execute(sqlstring, {
            'FBComment' : usrFBComment,
            'FBRating' : usrFBRating,
            'FBdatetime' : FBdatetime,
            'model': model,
            'brand': brand,
            'type': type
        })
    return redirect(url_for('home')) 


@app.route("/ndsbrandtype", methods=["GET", "POST"])
def ndsbrandtype():
    session["ndsbrands"]= str(request.form["ndsgadgetBrand"])
    temp_df = pd.read_sql("SELECT Distinct(Type) FROM gadget_reviews where Brand='" +session["ndsbrands"] +"'", sqlengine)
    gadgetType = temp_df["Type"].drop_duplicates()
    return render_template("newdataset.html", 
                           gadgetType = gadgetType.to_numpy(), 
                           selectbrand= session["ndsbrands"])
 
@app.route("/ndstypemodel", methods=["GET", "POST"])
def ndstypemodel():
    session["ndstype"]= str(request.form["gadgetType"])
    temp_df = pd.read_sql("SELECT Distinct(Model) FROM gadget_reviews where Brand='" +session["ndsbrands"] +"' and Type='"+session["ndstype"]+"'", sqlengine)
    gadgetModel = temp_df["Model"].drop_duplicates()
    return render_template("newdataset.html", 
                           gadgetModel = gadgetModel.to_numpy(), 
                           selectedtype = session["ndstype"], 
                           selectbrand= session["ndsbrands"])

@app.route("/ndsmodelcomplete", methods=["GET", "POST"])
def ndsmodelcomplete():
    session["ndsmodel"] = str(request.form["gadgetModel"])
    return render_template("newdataset.html", 
                           selectedtype = session["ndstype"], 
                           selectbrand= session["ndsbrands"], 
                           selectedmodel = session["ndsmodel"])

@app.route("/")
def home():
    temp_df = pd.read_sql("SELECT Distinct(Brand) FROM gadget_reviews order by Brand" , sqlengine)
    brands = temp_df["Brand"].drop_duplicates()
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
    gadgetType = temp_df["Type"].drop_duplicates().str.strip()
    return render_template("index.html", gadgetType = gadgetType.to_numpy(), selectbrand= session["brands"])
 
@app.route("/typemodel", methods=["GET", "POST"])
def typemodel():
    session["type"]= str(request.form["gadgetType"])
    temp_df = pd.read_sql("SELECT Distinct(Model) FROM gadget_reviews where Brand='" +session["brands"] +"' and Type='"+session["type"]+"' order by Model", sqlengine)
    gadgetModel = temp_df["Model"].drop_duplicates()
    return render_template("index.html", gadgetModel = gadgetModel.to_numpy(), selectedtype = session["type"], selectbrand= session["brands"])

@app.route("/modelcomplete", methods=["GET", "POST"])
def modelcomplete():
    session["model"] = str(request.form["gadgetModel"])
    return render_template("index.html", selectedtype = session["type"], selectbrand= session["brands"], selectedmodel = session["model"])

def sub_recommendation_summary(gadgetmodel):
    # gadgetmodel = 'AirPods'
    # global_table_name = "attrib_headset"
    mysqlconn.close()
    mysqlconn._open_connection()
    with sqlengine.begin() as connection:
        temp_df_count = pd.read_sql_query(sqlalch.text("SELECT count(model) as count FROM gadget_reviews where Model='"+gadgetmodel+"'"), connection)

    with sqlengine.begin() as connection:
        temp_df_reco = pd.read_sql(sqlalch.text(f"SELECT * FROM {global_table_name} where Model='"+gadgetmodel+"'"), connection)        
    
    if global_table_name == "attrib_headset":
        attrib_columns = ["Price", "Sound_Q", "Comfort", "Microphone", "Connectivity","Battery","NoiseCancellation","Design","Controls"]
    elif global_table_name == "attrib_smartphone":
        attrib_columns = ["Price","Battery","Camera","Performance", "Storage", "Display","OS", "Features"]
    elif global_table_name == "attrib_smartwatch":
        attrib_columns = ["Price", "Battery", "Design","Display","Health","Sports","Smart","Compatibility","Quality"]
    else: #Generic
        attrib_columns = ["Battery", "Screen", "Speed", "Memory", "Audio"]
    

    temp_df_reco['highest_value'] = temp_df_reco[attrib_columns].max(axis=1)
    temp_df_reco['highest_colname'] = temp_df_reco[attrib_columns].idxmax(axis=1)

    print ("Max value Value >> " + str(temp_df_reco['highest_value']))
    print ("Max value column name >> " + str(temp_df_reco['highest_colname']))

    head_featured = ""  
    sub_featured = ""
    highest_col_name = ''.join(temp_df_reco['highest_colname'])

    with sqlengine.begin() as connection:
        desc_message = pd.read_sql(sqlalch.text(f"SELECT * FROM featured_desc where Attrib_Name='"+highest_col_name+"'"), connection)        

    
    head_featured = ''.join(desc_message["Head_Featured_Desc"])
    sub_featured = ''.join(desc_message["Sub_Featured_Desc"])

    # summary_reco = [ temp_df_reco["Batt_PR"][0], temp_df_reco["Scr_PR"][0], temp_df_reco["Spd_PR"][0], temp_df_reco["Mem_PR"][0], temp_df_reco["Aud_PR"][0]] 
    temp_df_reco = temp_df_reco.drop(columns=['index', 'Model', 'highest_value','highest_colname'])
    melt_df_data = temp_df_reco.melt(var_name = 'Col_Names', value_name = 'Values')
    summary_reco = melt_df_data

    return summary_reco, head_featured, sub_featured 
    
def sub_AIresult(item_desc):
    # item_desc = "JBL T110BT"
    genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    airesult = str(model.generate_content("specifications of " + item_desc + " include reference with url. Show result in table body format source code only no comments from AI").text)
    airesult = airesult.replace("```html\n","")
    airesult = airesult.replace("\n```","")

    # airesult = airesult.replace("*","")
    return airesult

def sub_AIresult_Shop_Loc(item_desc):
    # item_desc = "JBL T110BT"
    genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    shoploc_list = str(model.generate_content( "list of stores to buy " + item_desc + " in the philippines. column arragement: Store, type, Price Range, URL reference. Show result in table body format source code only no comments from AI").text)
    shoploc_list = shoploc_list.replace("```html","")
    shoploc_list = shoploc_list.replace("\n```","")
    return shoploc_list
    
def sub_OpenAI(model, type, brand):
    default_img = "./static/HTML/images/NIA.jpg"
    query = text("""
        SELECT Path, Path2, Path3, Path4
        FROM image_paths
        WHERE model = :model AND type = :type AND brand = :brand
    """)

    with sqlengine.begin() as connection:
        result = connection.execute(query, {
            "model": model,
            "type": type,
            "brand": brand
        }).fetchone()

    if result:
        return result[0], result[1], result[2], result[3]
    else:
        return default_img, default_img, default_img, default_img
        
def sub_datacleaning(temp_df):
    # sqlstring = "SELECT * FROM gadget_reviews where Brand='" +brands+"' and Type='"+type+"' and Model='"+gadgetmodel + "'"
    # temp_df = pd.read_sql(sqlstring, sqlengine)
    lemmatizer = WordNetLemmatizer()

    #Remove Column Username since this column is unnecessary
    temp_df["Reviews"] = temp_df["Reviews"].str.lower()
        
    # Fill missing usernames
    temp_df["Username"] = temp_df["Username"].fillna("No Username")       
    
    # Date with invalid values will be default to 1/1/11, which also not useful :)
    temp_df["Date"] = temp_df["Date"].fillna("1/1/11")

    # All records with not value for REVIEWS will be dropped
    temp_df.dropna(subset=["Reviews"], inplace=True)
    temp_df = temp_df[temp_df["Reviews"].str.strip() != ""]

    # Replace all special characters into black spaces which will also be remove
    temp_df["Reviews"] = (
            temp_df["Reviews"]
            .str.replace(r'\n|\r', ' ', regex=True)
            .str.replace(r'http\S+', '', regex=True)
            .str.replace(r'x000D', '', regex=True)
            .str.replace(r'<[^>]+>', '', regex=True)
            .str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
            .str.replace(r'\s+[a-zA-Z]\s+', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
    )

    # Tokenize and lemmatize reviews
    # temp_df["Reviews"] = temp_df["Reviews"].apply(lambda text: ' '.join(lemmatizer.lemmatize(token) for token in word_tokenize(text)))
    temp_df['Reviews'] = temp_df['Reviews'].apply(lambda x: word_tokenize(x))
    
    def lemmatize_review(review_text):
        lemmatize_words = [lemmatizer.lemmatize(word) for word in review_text]
        lemmatize_text = ' '.join(lemmatize_words)
        return lemmatize_text
    temp_df['Reviews'] = temp_df['Reviews'].apply(lemmatize_review)
        

    # Drop any rows that became empty after processing
    temp_df.replace({"Reviews": {"": None}}, inplace=True)
    temp_df.dropna(subset=["Reviews"], inplace=True)
        
    # Convert Ratings to binary: 1 & 2 → 0 (Negative), 4 & 5 → 1 (Positive), drop 3 (Neutral)
    temp_df["Rating"] = temp_df["Rating"].astype(str)
    temp_df["Rating"] = temp_df["Rating"].str.replace(r'^[1-2]$', '0', regex=True)
    temp_df["Rating"] = temp_df["Rating"].str.replace(r'^[4-5]$', '1', regex=True)

    # Convert to int and drop neutral ratings
    temp_df["Rating"] = pd.to_numeric(temp_df["Rating"], errors='coerce')
    temp_df = temp_df[temp_df["Rating"].isin([0, 1])]

    return temp_df

def sub_generate_rating(dt_rating, lstm_rating, gadgetmodel):
    overall_rating = (dt_rating + lstm_rating) / 2

    scaled_rating = round(overall_rating * 5 * 2) / 2
    full_stars = int(scaled_rating)
    half_star = 1 if scaled_rating - full_stars == 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    visual_stars = '★' * full_stars + '½' * half_star + '☆' * empty_stars
    return (f"Estimated Star Rating: {scaled_rating} / 5  {visual_stars}"
    f"\n This gadget '{gadgetmodel} is ' {'Recommend' if overall_rating >= 0.5 else 'Not Recommend'} based on the system evaluation.")

def attrib_table(temp_df_attrib, gadgettype):
    #--------------------------------------------------------------------------------------------
    #Extracting phrases for creating corpora that will be use in data visualization
    # FF: temp_df_attrib here is a cleaned dataset came from datacleaning function
    #--------------------------------------------------------------------------------------------
    # temp_df_attrib = temp_df
    # gadgettype = "Ear Buds"
    df_reviews = temp_df_attrib.drop(axis=1, columns=["Date", "Rev_No", "Username"])
    df = pd.DataFrame()



    def extract_attrib(attrib_value):
        df_temp = df_reviews.loc[df_reviews["Reviews"].str.contains(attrib_value, regex=False)]
        df_temp["Reviews"] = df_temp["Reviews"].str.replace('[0-9]', "", regex=True)
        
        if attrib_value == "battery":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}battery\b(?:\W+\w+){0,2})')
        elif attrib_value == "speed":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}speed\b(?:\W+\w+){0,2})')
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}performance\b(?:\W+\w+){0,2})')
        elif attrib_value == "memory":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}memory\b(?:\W+\w+){0,2})')
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}RAM\b(?:\W+\w+){0,2})')
        elif attrib_value == "screen":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}display\b(?:\W+\w+){0,2})')
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}screen\b(?:\W+\w+){0,2})')
        elif attrib_value == "audio":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}audio\b(?:\W+\w+){0,2})')
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}sound\b(?:\W+\w+){0,2})')
        elif attrib_value == "comfort":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}comfort\b(?:\W+\w+){0,2})')
        elif attrib_value == "microphone":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}microphone\b(?:\W+\w+){0,2})')  
        elif attrib_value == "connectivity":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}connectivity\b(?:\W+\w+){0,2})')
        elif attrib_value == "design":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}design\b(?:\W+\w+){0,2})')
        elif attrib_value == "controls":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}controls\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "noisecancellation":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}noise cancel\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "price":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}price\b(?:\W+\w+){0,2})')
        elif attrib_value == "camera":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}camera\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "performance":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}performance\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "storage":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}storage\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "os":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}os\b(?:\W+\w+){0,2})')
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}operating system\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "features":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}features\b(?:\W+\w+){0,2})')
        elif attrib_value == "connectivity":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}connectivity\b(?:\W+\w+){0,2})')                              
        elif attrib_value == "health":
           df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}health\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "sports":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}sports\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "smart":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}smart\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "compatibility":
          df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}compatibility\b(?:\W+\w+){0,2})')                          
        elif attrib_value == "quality":
          df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}quality\b(?:\W+\w+){0,2})')                          
        else:
            df_temp["Attribute"] = attrib_value
        
        df_temp = df_temp.dropna(axis=0, subset=["Reviews"], how='any')
        df_temp = df_temp.drop_duplicates(subset="Reviews")
        df_temp["Attribute"] = attrib_value
        return df_temp
    print (gadgettype + " >> Gadget type")
    global list_attrib
    if gadgettype == "Ear Buds" or gadgettype == "Earphone":
        list_attrib = ["price", "audio", "comfort", "microphone", "connectivity", "battery","noisecancellation","design","controls" ]
    elif gadgettype == "Smartphone" or gadgettype == "Tablet":
        list_attrib = ["price","battery","camera","performance", "storage", "screen","OS"]
    elif gadgettype == "Smartwatch":
        list_attrib = ["price", "battery", "design","display","health","sports","smart","compatibility","quality"]
    else: #Generic
        list_attrib = ["battery", "screen", "speed", "memory", "audio"]

    for attrib in list_attrib:
        df = pd.concat([df, extract_attrib(attrib)])
    # df[df["Attribute"] == "memory"]
    # attrib_matrix = pd.DataFrame(columns=["Model", "Batt_PR", "Scr_PR", "Spd_PR", "Mem_PR", "Aud_PR"])
    if gadgettype == "Ear Buds" or gadgettype == "Earphone":
        attrib_matrix = pd.DataFrame(columns=["Model", "Price", "Sound_Q", "Comfort", "Microphone", "Connectivity","Battery","NoiseCancellation","Design","Controls"])
    elif gadgettype == "Smartphone" or gadgettype == "Tablet":
        attrib_matrix = pd.DataFrame(columns=["Model", "Price", "Battery", "Camera", "Performance", "Storage","Display","OS","Features"])
    elif gadgettype == "Smartwatch":
        attrib_matrix = pd.DataFrame(columns=["Model", "Price", "Battery", "Design", "Display", "Health","Sports","Smart","Compatibility", "Quality"])
    else: #Generic Gadget
        attrib_matrix = pd.DataFrame(columns=["Model", "Batt_PR", "Scr_PR", "Spd_PR", "Mem_PR", "Aud_PR"])
    gadget_list = df_reviews["Model"].unique()

    
    def convert_to_matrix(gadget_model):

        df_model = df.loc[df["Model"].str.contains(gadget_model)]
        df_rev = df_model.loc[df_model["Reviews"].str.contains("battery")]
        battery = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("screen")]
        screen = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("speed")]
        speed = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("memory")]
        memory = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("audio")]
        audio = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("comfort")]
        comfort = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("microphone")]
        microphone = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("connectivity")]
        connectivity = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("design")]
        design = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("controls")]
        controls = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("noisecancellation")]
        noisecancellation = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("price")]
        price = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("camera")]
        camera = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("performance")]
        performance = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("storage")]
        storage = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("os")]
        os = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("features")]
        features = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("connectivity")]
        connectivity = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("health")]
        health = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("sports")]
        sports = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("smart")]
        smart = df_rev["Rating"].value_counts().get(1,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("compatibility")]
        compatibility = df_rev["Rating"].value_counts().get(1,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("quality")]
        quality = df_rev["Rating"].value_counts().get(1,0)
        
        global global_table_name
        if gadgettype == "Ear Buds" or gadgettype == "Earphone":
            row_value= [gadget_model,price, audio, comfort, microphone, connectivity, battery,noisecancellation,design,controls]
            global_table_name = "attrib_headset"
        elif gadgettype == "Smartphone" or gadgettype == "Tablet":
            row_value = [gadget_model, price, battery, camera, performance, storage, screen, os, features]
            global_table_name = "attrib_smartphone"
        elif gadgettype == "Smartwatch":
            row_value = [gadget_model, price, battery, design, screen, health, sports, smart, compatibility, quality]
            global_table_name = "attrib_smartwatch"
        else:
            row_value = [gadget_model, battery, screen, speed, memory, audio]
        
        print(gadgettype + " >> gadget type inside def")
        print (global_table_name + " >> global table name INSIDE DEF")
        return row_value
    
    for colname in gadget_list:
        attrib_matrix.loc[len(attrib_matrix)] = convert_to_matrix(colname)

    print(gadget_list + " >> gadget list value")
    print (global_table_name + " >> global table name OUTSIDE DEF")
    attrib_matrix.to_sql(con=sqlengine, name= global_table_name, if_exists='replace', index=True)

def attrib_graph(data_record):
    # data_record = summary_reco
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    plt.figure(figsize=(10,6))
    plt.barh(data_record['Col_Names'], data_record['Values'], color='Blue')
    plt.xlabel('No. of User Reviews')
    plt.title('Summary of User Gadget Reviews')
    plt.savefig("./static/HTML/images/Summary_Graph.png")
    plt.close()

@app.route("/otherselection", methods=["GET","POST"])
def sub_other_recommendation():
    gadgetmodel = str(request.form["otherModel"])
    gadgetmodel = "iPhone 14"
    sqlstring = f"SELECT Brand, Type, Model FROM gadget_reviews WHERE Model = '{gadgetmodel}'"
    with sqlengine.begin() as sqlconnection:
        temp_result = pd.read_sql(sqlalch.text(sqlstring), sqlconnection)
        brandtype = str(temp_result['Brand'][0])
        type = temp_result['Type'][0]
        gadgetmodel = temp_result['Model'][0]      
        # modelrecommendation()
    return render_template("testvalue.html", value1 = session['Brand'], value2 = session['Type'], value3=session['Model']) 
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
    # cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax = sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues")   
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
    try:
        gadgettype = "Smartphone"
        kmeans_df = pd.read_sql("SELECT Rev_No, Model, Rating FROM gadget_reviews where Type='" + gadgettype + "'", sqlengine)
        kmeans_df = kmeans_df.iloc[:10000,:]
        df_reco = kmeans_df[["Rev_No",'Model', 'Rating']]
        pivot_table = pd.pivot_table(df_reco, index='Rev_No', columns="Model", values='Rating', fill_value=0)
        
        num_clusters = 3
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
    except:
        k=0
        top_kmeans_reco = ("No other gadget to recommend","1")
        return top_kmeans_reco, k

def sub_decision_tree(gadgettype):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    dectree_df = pd.read_sql("SELECT Brand, Type, Model, Reviews, Rating FROM gadget_reviews", sqlengine)
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

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Precision: {precision:.3%}")
    print(f"Recall: {recall:.3%}")
    print(f"F1 Score: {f1:.3%}")

if __name__ == "__main__":
    app.run(debug=True)
