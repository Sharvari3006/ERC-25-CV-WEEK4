import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# 1. Use a live feed from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously detect hands
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # 3. Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Continuously detect hands
    results = hands.process(frame_rgb)

    # 2. Draw the detected hand landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS) # This draws both landmarks and connections

    # Display the frame
    cv2.imshow('MediaPipe Hand Detection', frame)

    # 4. Stop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release and cleanup
cap.release()
cv2.destroyAllWindows()