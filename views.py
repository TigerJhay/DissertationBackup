from flask import Blueprint, render_template, request, flash, jsonify
import google.generativeai as genai
import tensorflow as tf
import os
import pandas as pd
import numpy as np

# from HTMLparser import HTMLParser


views = Blueprint(__name__, "views")

@views.route("/")
def home():
     return render_template("index.html")
     #return render_template("testscript.html")

@views.route("/generateResult", methods=["GET","POST"])
def fetchAIdesc():
     searchstring = str(request.form['txtsearch'])
     genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
     model = genai.GenerativeModel("gemini-1.5-flash")
     response = model.generate_content(searchstring)

     #response.text
     flash("***Updated Generated AI response: " + str(response.text.replace("**","\n")))
     return render_template("index.html")

@views.route("/testroute")
def testarea(): 
     
     # df = pd.read_csv(csv_path)
     # df 
     return render_template("testscript.html")


@views.route("/testflask")
def testarea2():
     strmessage1 = "This is text \n inside flash"
     strmessage2 = "Lorem ipsum dolor sit amet ** , consectetur adipiscing elit **"
     
     strmessage3 = "Lorem ipsum dolor sit amet ** <br> , consectetur adipiscing elit ** , sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
     #print(strmessage3)
     return render_template("testscript.html", message_content = strmessage3)
     
@views.route("/search_gadget_description", methods=["GET","POST"])
def fetchAIdescription():
     searchstring = str(request.form['txtSearchValue'])
     genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
     model = genai.GenerativeModel("gemini-1.5-flash")
     response = model.generate_content(searchstring)    
     flash("Generated AI response: " + str(response.text))
     return render_template("testscript.html")

@views.route("/profile")
def profile():
    args = request.args
    name = args.get('name')
    return render_template("index.html", name=name)