###################################################################################################################################################################
#  @Author:- Sagar Sharma
#  @Title:- Basic Hand Recognition using Mediapipe 
###################################################################################################################################################################

import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import numpy as np
import time
import tensorflow as tf
from pynput.keyboard import Key, Controller
import subprocess

# Initialize webcam
cap = cv2.VideoCapture(0)
x, y = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, x)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, y)

# Initialize MediaPipe Hand module
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize pycaw for audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Variables for hand gesture detection and control
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]  # Finger coordinates for gesture recognition
thumb_Coord = (4, 2)  # Thumb coordinates
volbar = 700  # Initial volume bar position
volper = 0  # Initial volume percentage
blbar = 400  # Initial brightness bar position
blper = 0  # Initial brightness percentage
volMin, volMax = volume.GetVolumeRange()[:2]  # Volume range
blMin = 10  # Minimum brightness
blMax = 100  # Maximum brightness
ptime = 0  # Previous time for FPS calculation
newTime = 0  # Current time for FPS calculation
timer = 0  # Timer for gesture cooldown
counting = False  # Flag for gesture counting
cooldown = 0  # Cooldown timer for gestures
direction = "none"  # Direction of swipe gesture
keyboard = Controller()  # Controller for keyboard inputs
flag = 0  # Flag for swipe direction display
zoom_cooldown = 0  # Initialize zoom cooldown
zooming = False  # Flag to track zooming state

while True:
    ret, img = cap.read()  # Capture frame from webcam
    if not ret:
        break

    img = cv2.flip(img, 1)  # Flip the frame horizontally
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

    newTime = time.time()
    if counting:
        timer += newTime - ptime

    fps = 1 / (newTime - ptime)  # Calculate FPS
    ptime = newTime
    img.flags.writeable = False
    results = hands.process(imgRGB)  # Process the frame with MediaPipe
    img.flags.writeable = True
    cv2.line(img, (400, 0), (400, 728), (255, 0, 255), thickness=2)  # Draw vertical line for gesture detection
    cv2.line(img, (x - 1000, 0), (x - 1000, 728), (255, 0, 255), thickness=2)  # Draw another vertical line
    cv2.putText(img, f'timer: {int(timer)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)  # Display timer
    cv2.putText(img, f'cooldown: {int(cooldown)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)  # Display cooldown

    # Collection of gesture information
    results = hands.process(imgRGB)  # Process the frame again if necessary
    lmList = []  # List to store hand landmarks

    if results.multi_hand_landmarks:  # Check if any hands are detected
        handList = []
        for handlandmark in results.multi_hand_landmarks:  # Iterate over detected hands
            for id, lm in enumerate(handlandmark.landmark):  # Get each landmark's ID and coordinates
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel coordinates
                handList.append((cx, cy))
                lmList.append([id, cx, cy])  # Append landmark information to list
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)  # Draw hand landmarks
            for point in handList:
                cv2.circle(img, point, 5, (255, 255, 255), cv2.FILLED)  # Draw circles at landmark positions

    if lmList:
        # Get coordinates of various fingers
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
        x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger
        x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger
        x5, y5 = lmList[20][1], lmList[20][2]  # Pinky finger

        # Draw circles at finger tips
        cv2.circle(img, (x1, y1), 15, (67, 168, 25), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (67, 222, 255), cv2.FILLED)
        cv2.circle(img, (x4, y4), 15, (111, 200, 155), cv2.FILLED)
        cv2.circle(img, (x5, y5), 15, (137, 20, 150), cv2.FILLED)

        # Detect specific gestures
        fingers = [False] * 5
        if handList[4][0] < handList[3][0]: fingers[0] = True  # Thumb is up
        if handList[8][1] < handList[6][1]: fingers[1] = True  # Index finger is up
        if handList[12][1] < handList[10][1]: fingers[2] = True  # Middle finger is up
        if handList[16][1] < handList[14][1]: fingers[3] = True  # Ring finger is up
        if handList[20][1] < handList[18][1]: fingers[4] = True  # Pinky finger is up
        

        # Check if only index and middle fingers are open for zoom operations
        if fingers == [False, True, True, False, False]:
            zooming = True  # Set zooming flag
            # Calculate the distance between the index and middle fingers
            length = hypot(x3 - x2, y3 - y2)

            # Define the range for zooming
            zoom_in_threshold = 60  # Minimum distance for zoom in
            zoom_out_threshold = 40  # Maximum distance for zoom out

            # Slow zoom in
            if length > zoom_in_threshold:
                print("Zooming in")
                keyboard.press(Key.ctrl)
                keyboard.press('=')
                keyboard.release('=')
                keyboard.release(Key.ctrl)

            # Slow zoom out
            elif length < zoom_out_threshold:
                print("Zooming out")
                keyboard.press(Key.ctrl)
                keyboard.press('-')
                keyboard.release('-')
                keyboard.release(Key.ctrl)

            # Display zoom level on the frame
            cv2.putText(img, f'Zoom Level: {int(length)}', (980, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, 'Peace', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        else:
            zooming = False  # Reset zooming flag
            # Handle other gestures
            if fingers == [False, True, True, False, False]:
                cv2.putText(img, 'Peace', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            elif fingers == [True, False, False, False, False]:
                cv2.putText(img, 'Thumbs Up', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            elif fingers == [False, False, True, True, True]:
                cv2.putText(img, 'OK', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            elif fingers == [False, True, False, False, True]:
                cv2.putText(img, 'Rock On', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
            elif fingers == [True, True, True, True, True]:
                cv2.putText(img, 'Hi', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                keyboard.press(Key.media_play_pause)  # Toggle play/pause

        """
        elif fingers == [False, False, True, False, False]:
            cv2.putText(img, 'Middle Finger', (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            subprocess.call(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"])  # Sleep mode command
        """

        # Detect swipe gestures
        if (x3 < 400 or x4 < 400 or x5 < 400) and cooldown == 0 and direction == "none":
            counting = True
            timer = 0
            direction = "right"

        if (x3 > x - 1000 or x4 > x - 1000 or x5 > x - 1000) and cooldown == 0 and direction == "none":
            counting = True
            timer = 0
            direction = "left"

        # Check if swipe right gesture is detected
        if (x3 > x - 1000 or x4 > x - 1000 or x5 > x - 1000) and timer < 1.5 and direction == "right":
            print("Swiped right")
            keyboard.press(Key.left)  # Simulate left arrow key press
            keyboard.release(Key.left)
            cooldown = 5
            timer = 0
            direction = "none"
            flag = 4

        # Check if swipe left gesture is detected
        if (x3 < 400 or x4 < 400 or x5 < 400) and timer < 1.5 and direction == "left":
            print("Swiped left")
            keyboard.press(Key.right)  # Simulate right arrow key press
            keyboard.release(Key.right)
            timer = 0
            cooldown = 5
            direction = "none"
            flag = 2

        # Handle hand classification
        if len(results.multi_handedness) != 2:
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)  # Draw circle for left hand
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw line between thumb and index finger
                    cv2.putText(img, label + ' Hand', (20, 280), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  # Display left hand label

                    upCount = 0
                    for coordinate in finger_Coord:
                        if handList[coordinate[0]][1] < handList[coordinate[1]][1]:  # Count fingers that are up
                            upCount += 1
                    if handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]:
                        upCount += 1
                    cv2.putText(img, f'Finger Count: {str(upCount)}', (540, 80), cv2.FONT_HERSHEY_PLAIN, 2, (133, 15, 59), 4)  # Display finger count

                    if (upCount == 2):
                        length = hypot(x2 - x1, y2 - y1)  # Distance between thumb and index finger
                        vol = np.interp(length, [50, 420], [volMin, volMax])  # Map distance to volume level
                        volbar = np.interp(length, [50, 700], [700, 150])  # Map distance to volume bar position
                        volper = np.interp(length, [50, 400], [0, 100])  # Map distance to volume percentage
                        print(vol, int(length))
                        volume.SetMasterVolumeLevel(vol, None)  # Set volume level

                if label == 'Right':
                    cv2.circle(img, (x2, y2), 15, (128, 255, 56), cv2.FILLED)  # Draw circle for right hand
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw line between thumb and index finger
                    cv2.putText(img, label + ' Hand', (980, 280), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  # Display right hand label

                    upCount = 0
                    for coordinate in finger_Coord:
                        if handList[coordinate[0]][1] < handList[coordinate[1]][1]:  # Count fingers that are up
                            upCount += 1
                    if handList[thumb_Coord[0]][0] < handList[thumb_Coord[1]][0]:
                        upCount += 1
                    cv2.putText(img, f'Finger Count: {str(upCount)}', (540, 80), cv2.FONT_HERSHEY_PLAIN, 2, (133, 15, 59), 4)  # Display finger count

                    if (upCount == 2):
                        length1 = hypot(x2 - x1, y2 - y1)  # Distance between thumb and index finger
                        bl = np.interp(length1, [10, 100], [blMin, blMax])  # Map distance to brightness level
                        blbar = np.interp(length1, [30, 100], [100, 30])  # Map distance to brightness bar position
                        blper = np.interp(length1, [10, 100], [0, 100])  # Map distance to brightness percentage

                        sbc.set_brightness(bl)  # Set screen brightness

        # Display both hands message if both hands are detected
        else:
            cv2.putText(img, 'Both Hands', (540, 80), cv2.FONT_HERSHEY_PLAIN, 2, (133, 15, 59), 4)

        # Creating volume bar for volume level
        cv2.rectangle(img, (50, 400), (85, 700), (255, 0, 0), 4)  # Draw volume bar border
        cv2.rectangle(img, (50, int(volbar)), (85, 700), (255, 0, 0), cv2.FILLED)  # Fill volume bar
        cv2.putText(img, f"Volume: {int(volper)}%", (20, 320), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)  # Display volume percentage

        cv2.putText(img, f"Brightness: {int(blper)}%", (980, 320), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)  # Display brightness percentage

    # Display swipe direction text
    if (flag == 4):
        cv2.putText(img, f'Swiped right', (20, 360), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow("Image", img)
    elif (flag == 2):
        cv2.putText(img, f'Swiped left', (980, 360), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow("Image", img)
    else:
        cv2.imshow("Image", img)  # Display the video

    # Reset timer and direction if necessary
    if timer >= 1.5:
        timer = 0
        counting = False
        direction = "none"

    # Decrease cooldown timer if it is active
    if cooldown > 0:
        cooldown -= 1

    # Break loop if 'q' key is pressed
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()  # Release webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
