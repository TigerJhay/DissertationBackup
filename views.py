from flask import Blueprint, render_template, request, flash, jsonify
import google.generativeai as genai

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
     #temporary_value = "The quick brown fox jumps over the lazy dog"
     flash("Generated AI response: " + str(response.text))
     #jsonify(result = temporary_value)
     return render_template("index.html")

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

# /get_genAI is URL path of HTML
@views.route("/get_genAI")
def genAI():
     flash ("Hello Tiger, This is from flash message")
     return render_template("testscript.html")

#'passvalue' should be the action in <form action=''
@views.route("/passvalue",methods=["POST", "GET"])
def passval():
     flash("Value pass is: " + str(request.form['txtSearchValue']))
     return render_template("testscript.html") 




