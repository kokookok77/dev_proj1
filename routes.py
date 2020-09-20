from flask import Blueprint, request, jsonify

bp = Blueprint("image_bp", __name__, url_prefix="/imgInput", template_folder='templates')

@bp.route('/imageUpload', methods=['POST'])
def imageUpload():
    '''
        Image Upload
    '''

    try:
        image = request.form['image_input']
        print(image)

    except Exception as e:
        print('Error:', e)

    return ""