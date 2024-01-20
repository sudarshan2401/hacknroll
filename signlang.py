import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def sign_language(curr):
    cap = cv2.VideoCapture(0)
    # Setup mediapipe instance
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection wrd
            results = hands.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if multiple hands
            if results.multi_hand_landmarks:
                # Render detections
                for hand_landmarks in results.multi_hand_landmarks:
                    handSign = getHandSign(hand_landmarks)
                    print(curr)
                    if handSign == curr:
                        print("mf works")
                    # print(getHandSign(hand_landmarks))
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

            # cv2.imshow('Mediapipe Feed', image)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def getHandSign(hand_landmarks) :
    thumb_finger_tip = hand_landmarks.landmark[4]
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    ring_finger_tip = hand_landmarks.landmark[16]
    pinky_finger_tip = hand_landmarks.landmark[20]

    # Check if hand is in OK gesture
    if thumb_finger_tip.y < index_finger_tip.y < middle_finger_tip.y < ring_finger_tip.y < pinky_finger_tip.y:
        return "thumbs_up"

    elif thumb_finger_tip.y > index_finger_tip.y > middle_finger_tip.y > ring_finger_tip.y > pinky_finger_tip.y:
        return "thumbs_down"
    # Check if hand is in Stop gesture
    elif thumb_finger_tip.x > index_finger_tip.x > middle_finger_tip.x:
        if (hand_landmarks.landmark[2].x > hand_landmarks.landmark[5].x) and (
                hand_landmarks.landmark[3].x > hand_landmarks.landmark[5].x) and (
                hand_landmarks.landmark[4].x > hand_landmarks.landmark[5].x):
            return "stop"
        else:
            return "Not recognised"
    else:
        return "Not recognised"
