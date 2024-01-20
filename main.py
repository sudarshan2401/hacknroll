from flask import Flask, render_template, Response
from signlang import sign_language

app = Flask(__name__)

@app.route('/')
@app.route('/index')

def index():
    return render_template('index.html')

def generate_frames():
    yield from sign_language()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)