from flask import Flask
from views import views

#need this line to access HTML files inside templates folder
app = Flask(__name__)
app.register_blueprint(views, url_prefix = "/")
#this secret_key does not matter, it is just from avoiding error during execution
app.secret_key = "abcdef12345"
if __name__ == "__main__":
    app.run(debug=True)

