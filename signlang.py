import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the score variable outside the function
score = 0

def sign_language(curr):
    global score  # Access the global score variable
    cap = cv2.VideoCapture(0)
    # Setup mediapipe instance
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        prev_sign = None
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = hands.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if multiple hands
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_sign = getHandSign(hand_landmarks)
                    print(curr)
                    if hand_sign == curr and hand_sign != prev_sign:
                        cv2.putText(image, f"Correct Sign: {curr}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10, cv2.LINE_AA)
                        print("Correct sign detected: ", curr)
                        score += 1
                        prev_sign = curr

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

            # Display the score on the frame
            cv2.putText(image, f"Score: {score}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6, cv2.LINE_AA)

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
