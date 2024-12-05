import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Key, Controller

# Set the desired width and height for the webcam screen
width, height = 640, 480

# Open the webcam and set the width and height for capturing video frames
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Initialize the HandDetector with a confidence threshold and max no. of hands to detect
detector = HandDetector(detectionCon=0.7, maxHands=1)

# Create a keyboard controller to simulate key presses
keyboard = Controller()

try:
    while True:
        # Read a frame from the webcam
        ret, img = cap.read()

        if not ret:
            print("Failed to read from the camera.")
            break

        # Detect the hand in the frame
        hands = detector.findHands(img, draw=True)  # Detect hands and draw landmarks

        if hands:  # Check if any hand is detected
            hand = hands[0]  # Get the first detected hand (a dictionary)
            if 'lmList' in hand:
                lmList = hand['lmList']  # List of landmarks (21 points)

                if len(lmList) > 20:  # Ensure all landmarks are detected
                    # Get the tip positions of the fingers
                    fingertips_y = np.array([lmList[i][1] for i in [4, 8, 12, 16, 20]])  # Y-coordinates of tips
                    finger_bases_y = np.array([lmList[i - 2][1] for i in [4, 8, 12, 16, 20]])  # Y-coordinates of bases

                    # Determine which fingers are up
                    fingers_up = fingertips_y < finger_bases_y

                    # Trigger actions based on finger gestures
                    if fingers_up[1]:  # Index finger up
                        keyboard.press(Key.right)
                        print("Right arrow key pressed")
                    else:
                        keyboard.release(Key.right)

                    if fingers_up[0]:  # Thumb up
                        keyboard.press(Key.left)
                        print("Left arrow key pressed")
                    else:
                        keyboard.release(Key.left)

        else:
            # If no hand is detected, release the keys
            keyboard.release(Key.left)
            keyboard.release(Key.right)

        # Show the image with hand gesture information
        cv2.imshow("Hand Gesture Control", img)

        # Check for the "q" key press to exit the infinite loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("Exception:", e)

finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
