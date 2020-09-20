import os

from flask import Blueprint, request, jsonify, make_response, redirect, url_for
from werkzeug.utils import secure_filename

bp = Blueprint("image_bp", __name__, url_prefix="/imgInput", template_folder='templates')


ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

@bp.route('/imageUpload', methods=["POST"])
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
            file.save(os.path.join('./', filename))

        return ""

            # if file and allowed_file(file.filename):
            #     file.save(os.path.join('./', filename))
            #     print(file.filename)
            # else:
            #     print('Error: "File type is not allowed"')

    except Exception as e:
        print('Error:', e)

        return ""