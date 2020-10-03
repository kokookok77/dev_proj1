import os

from flask import Flask, render_template, Blueprint, request
from flask import jsonify, make_response, redirect, url_for
from werkzeug.utils import secure_filename

from python_code.utils import allowed_file

import config

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route("/profile")
@app.route("/")
def profile():
    return render_template("profile.html")


@app.route('/imageUpload', methods=["POST"])
def imageUpload():
    '''
        Image Upload
    '''

    try:
        files = request.files.getlist("uploadImage")
        print(files)
        for file in files:
            print(file)
            filename = secure_filename(file.filename)
            file.save(os.path.join(config.img_save_path, filename))

        return
        # if file and allowed_file(file.filename):
        #     file.save(os.path.join('./', filename))
        #     print(file.filename)
        # else:
        #     print('Error: "File type is not allowed"')

    except Exception as e:
        print('Error:', e)

        return ""


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
