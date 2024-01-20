from flask import Flask, render_template, Response
from waitress import serve
from signlang import sign_language
import cv2

app = Flask(__name__)

imageNames = [
    "thumbs_up",
    "thumbs_down",
    "stop"
]

counter = 0

@app.route('/')
@app.route('/index')

def index():
    return render_template('index.html')

def generate_frames():
    yield from sign_language()

@app.route('/button/')
def button():
    global counter
    curr = imageNames[counter]
    counter = (counter + 1) % len(imageNames)
    return curr
    


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)