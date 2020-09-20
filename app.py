from flask import Flask, render_template, Blueprint, request
from routes import bp as image_bp


app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True

app.register_blueprint(image_bp)

@app.route("/profile")
@app.route("/")
def profile():
    return render_template("profile.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)