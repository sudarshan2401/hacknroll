from flask import Flask, render_template, Response
from signlang import sign_language

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

def generate_frames(curr):
    yield from sign_language(curr)

@app.route('/button/')
def button():
    global counter
    curr = imageNames[counter]
    counter = (counter + 1) % len(imageNames)
    return curr
    


@app.route('/video_feed/<curr>')
def video_feed(curr):
    return Response(generate_frames(curr), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)