# SignLingo

SignLingo is a web application that uses computer vision to recognize hand signs using the MediaPipe library. The application allows users to interact with sign language gestures, providing a video feed with real-time hand sign detection.

## Features

- Real-time hand sign detection using the MediaPipe library.
- Dynamic scoring system to track the user's performance in recognizing signs.
- Interactive web interface with a video feed and a changing set of hand signs.

## Requirements

- Python 3.6 or later
- Flask
- OpenCV
- MediaPipe

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/SignLingo.git
cd SignLingo
```
2. Install the required Python packages:

```
pip install -r requirements.txt
```
3. Run the Flask application:

```
python main.py
```

4. Visit http://localhost:8000 in your web browser to access SignLingo.

## Usage

    Upon accessing the web interface, you will see a video feed capturing your hand gestures.
    The word to be signed will change periodically.
    Click the "Next" button to change the word, and the application will provide feedback on correct sign recognition.
    The score is dynamically updated, and you can track your performance.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

    MediaPipe library: https://github.com/google/mediapipe
    https://www.youtube.com/watch?v=06TE_U21FK4&t=1684s

## Authors

Sudarshan

    GitHub: github.com/sudarshan2401
    LinkedIn: linkedin.com/in/kotasudarshankaranth

Nicholas

    Github: github.com/nicholascher
