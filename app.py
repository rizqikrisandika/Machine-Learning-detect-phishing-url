<<<<<<< HEAD
import os
import phishing_detection
from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from flask import jsonify
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/result')
def result():
    urlname  = request.args['name']
    result  = phishing_detection.getResult(urlname)
    return result

@app.route('/', methods = ['GET', 'POST'])
def hello():
	return  render_template("index.html")

			


if __name__ == '__main__':
=======
import os
import phishing_detection
from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from flask import jsonify
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/result')
def result():
    urlname  = request.args['name']
    result  = phishing_detection.getResult(urlname)
    return result

@app.route('/', methods = ['GET', 'POST'])
def hello():
	return  render_template("index.html")

			


if __name__ == '__main__':
>>>>>>> add112012562178373980aeafff718f900fe4b31
    app.run(debug=True)