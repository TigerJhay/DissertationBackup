from flask import Blueprint, render_template, request, flash, jsonify
import google.generativeai as genai

views = Blueprint(__name__, "views")

@views.route("/")
def home():
     return render_template("index.html")
     #return render_template("testscript.html")


@views.route("/testflask")
def testarea2():
     strmessage1 = "This is text \n inside flash"
     strmessage2 = "Lorem ipsum dolor sit amet ** , consectetur adipiscing elit **"
     strmessage3 = "Lorem ipsum dolor sit amet ** <br> , consectetur adipiscing elit ** , sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
     return render_template("testscript.html", message_content = strmessage3)
     
@views.route("/testroute")
def testarea():
     strmessage = "Lorem ipsum dolor sit amet ** , consectetur adipiscing elit ** , sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
     flash(strmessage)
     return render_template("testscript.html")

@views.route("/generateResult", methods=["GET","POST"])
def fetchAIdesc():
     searchstring = str(request.form['txtsearch'])
     genai.configure(api_key="AIzaSyDgRaOiicnXJSx_     GNtfvuNxKLhCDCDpHhQ")
     model = genai.GenerativeModel("gemini-1.5-flash")
     response = model.generate_content(searchstring)

     #response.text
     flash("Generated AI response: " + str(response.text.replace("**","\n")))
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




