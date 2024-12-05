import cv2
import time
import mediapipe as mp
import pyautogui
import math
import numpy as np
from pynput.keyboard import Controller
from tkinter import Tk, Button, Label

# Mediapipe Hand Detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

# Keyboard Controller
keyboard = Controller()

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
screen_width, screen_height = pyautogui.size()

# Virtual Keyboard Buttons
class ButtonObj:
    def __init__(self, pos, text, size=[70, 70]):
        self.pos = pos
        self.size = size
        self.text = text

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "CL"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "SP"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "APR"]]

button_list = [ButtonObj([100 * j + 10, 100 * i + 10], key) for i in range(len(keys)) for j, key in enumerate(keys[i])]

# Draw Keyboard Buttons
def draw_keyboard_buttons(img, button_list):
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (96, 96, 96), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img

# Distance Calculation
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Main Menu
def main_menu():
    root = Tk()
    root.title("Select Mode")
    root.geometry("400x300")
    root.configure(bg="#2E3B4E")

    def set_mode_to_mouse():
        root.destroy()
        mouse_mode()

    def set_mode_to_keyboard():
        root.destroy()
        keyboard_mode()

    Label(root, text="Hand Gesture Controller", font=("Arial", 18, "bold"), bg="#2E3B4E", fg="white").pack(pady=20)

    Button(root, text="Mouse Control", command=set_mode_to_mouse, width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)
    Button(root, text="Keyboard Control", command=set_mode_to_keyboard, width=20, height=2, bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=10)

    root.mainloop()

# Mouse Control Mode
# Mouse Control Mode
def mouse_mode():
    last_click_time = time.time()
    double_click_interval = 0.5  # Max time between clicks to be considered a double-click
    is_right_click = False

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpdraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                landmarks = [(lm.x * frame_width, lm.y * frame_height) for lm in hand_landmarks.landmark]
                
                # Pinch Gesture (thumb and index finger touch)
                thumb_x, thumb_y = int(landmarks[4][0]), int(landmarks[4][1])
                index_x, index_y = int(landmarks[8][0]), int(landmarks[8][1])
                distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

                if distance < 50:  # Pinch threshold (right-click)
                    pyautogui.click(button='right')  # Perform right-click
                    time.sleep(0.5)  # Add a delay to prevent multiple right-clicks in quick succession

                # Double-click Gesture (index and middle fingers touch)
                index_x, index_y = int(landmarks[8][0]), int(landmarks[8][1])
                middle_x, middle_y = int(landmarks[12][0]), int(landmarks[12][1])
                double_click_distance = calculate_distance(index_x, index_y, middle_x, middle_y)

                if double_click_distance < 50:  # Double-click threshold
                    if time.time() - last_click_time < double_click_interval:  # Double-click detection
                        pyautogui.doubleClick()  # Perform double-click
                        last_click_time = time.time()  # Reset the time after double-click
                    else:
                        last_click_time = time.time()  # Update the time for single click

                # Moving the cursor with the index finger
                if landmarks:
                    index_x, index_y = int(landmarks[8][0]), int(landmarks[8][1])
                    pyautogui.moveTo(screen_width / frame_width * index_x, screen_height / frame_height * index_y)

        cv2.putText(frame, "Press 'B' to return to Main Menu", (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.imshow("Mouse Control", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('b'):
            main_menu()  # Go back to the main menu
            break

    cap.release()
    cv2.destroyAllWindows()


# Keyboard Control Mode
def keyboard_mode():
    text = ""
    delay = 0
    capitalize = False  # Capitalization mode is off by default
    typing_allowed = False  # Flag to allow typing when two fingers touch

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        frame = draw_keyboard_buttons(frame, button_list)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpdraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                landmarks = [(lm.x * frame_width, lm.y * frame_height) for lm in hand_landmarks.landmark]

                # Check for pinch gesture (two fingers touching)
                thumb_x, thumb_y = int(landmarks[4][0]), int(landmarks[4][1])
                index_x, index_y = int(landmarks[8][0]), int(landmarks[8][1])
                distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

                if distance < 50:  # Pinch threshold, adjust as needed
                    typing_allowed = True  # Enable typing once pinch is detected
                else:
                    typing_allowed = False  # Disable typing if fingers are apart

                # If typing is allowed, detect key presses
                if typing_allowed and delay == 0:
                    for button in button_list:
                        x, y = button.pos
                        w, h = button.size
                        if x < index_x < x + w and y < index_y < y + h:
                            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 255, 255), cv2.FILLED)
                            cv2.putText(frame, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                            if delay == 0:
                                if button.text == "SP":
                                    text += " "
                                elif button.text == "CL":
                                    text = text[:-1]
                                elif button.text == "APR":
                                    capitalize = not capitalize  # Toggle capitalization mode
                                else:
                                    # Check if we are in capitalize mode and type accordingly
                                    letter = button.text
                                    if capitalize:
                                        letter = letter.upper()
                                    text += letter
                                    keyboard.press(letter)
                                delay = 1

        if delay > 0:
            delay += 1
            if delay > 10:
                delay = 0

        cv2.rectangle(frame, (20, 400), (900, 500), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (30, 470), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        cv2.putText(frame, "Press 'B' to return to Main Menu", (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.imshow("Keyboard Control", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('b'):
            main_menu()  # Go back to the main menu
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_menu()
