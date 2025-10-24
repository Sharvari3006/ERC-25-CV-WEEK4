import cv2
import mediapipe as mp
import numpy as np

# --- Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, # Only track one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get frame dimensions
success, temp_frame = cap.read()
if not success:
    print("Failed to read from webcam")
    cap.release()
    exit()
    
h, w, c = temp_frame.shape

# Create a blank canvas to draw on
canvas = np.zeros((h, w, 3), np.uint8)

# Initialize drawing variables
draw_color = (0, 0, 255)  # Default color: RED
thickness = 10
xp, yp = 0, 0  # Previous (x, y) coordinates

print("Starting Simple Drawing Pad... Press 'q' to quit.")
print("--- Controls ---")
print("'r' = RED | 'g' = GREEN | 'b' = BLUE")
print("'e' = ERASER | 't' = THICKNESS | 'c' = CLEAR")

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get coordinate for index finger tip (Landmark 8)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        cx, cy = int(index_tip.x * w), int(index_tip.y * h) # Current x, y

        # --- This meets the "Moving the finger... draws" requirement ---
        # It will *always* draw if a hand is seen.
        
        # On the first frame, just set the previous point
        if xp == 0 and yp == 0:
            xp, yp = cx, cy

        # Draw the line on the canvas
        # Use a large thickness for the eraser
        current_thickness = 50 if draw_color == (0, 0, 0) else thickness
        cv2.line(canvas, (xp, yp), (cx, cy), draw_color, current_thickness)
        
        # Update the previous point
        xp, yp = cx, cy

    else:
        # If no hand is detected, reset the previous point.
        # This "lifts the pen" when you move your hand away.
        xp, yp = 0, 0

    # --- Merge the Canvas and the Frame ---
    # This is the simplest way to blend the two images
    # Create an inverse mask of the canvas (where it's black)
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Create the main mask (where it's not black)
    mask = cv2.bitwise_not(mask_inv)
    
    # "Cut out" the drawing area from the main frame
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Get the drawing from the canvas
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    
    # Add them together
    result_frame = cv2.add(frame_bg, canvas_fg)

    # Display the current color/tool
    tool_text = "COLOR: RED"
    if draw_color == (0, 255, 0):
        tool_text = "COLOR: GREEN"
    elif draw_color == (255, 0, 0):
        tool_text = "COLOR: BLUE"
    elif draw_color == (0, 0, 0):
        tool_text = "ERASER"
    cv2.putText(result_frame, tool_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(result_frame, f"THICKNESS: {thickness}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    # Display the final frame
    cv2.imshow('Simple CV Drawing Pad', result_frame)

    # --- Handle Keyboard Commands ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        draw_color = (0, 0, 255) # RED
    elif key == ord('g'):
        draw_color = (0, 255, 0) # GREEN
    elif key == ord('b'):
        draw_color = (255, 0, 0) # BLUE
    elif key == ord('e'):
        draw_color = (0, 0, 0) # ERASER (draws black)
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), np.uint8) # CLEAR
    elif key == ord('t'):
        if thickness == 10:
            thickness = 20
        elif thickness == 20:
            thickness = 30
        else:
            thickness = 10

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()