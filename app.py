import os

from flask import Flask, render_template, Blueprint, request
from flask import jsonify, make_response, redirect, url_for
from werkzeug.utils import secure_filename
from routes import bp as image_bp

from python_code.utils import allowed_file

import config

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.register_blueprint(image_bp)

@app.route("/profile")
@app.route("/")
def profile():
    return render_template("profile.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
