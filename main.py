import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector with lower detection confidence
detector = HandDetector(detectionCon=0.6, maxHands=1)

# Find Function
# x is the raw distance, y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

# Loop
while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)  # Get hand data

    # Debugging: Print if hands are detected
    if hands:
        print("Hands detected!")

        hand = hands[0]  # Get the first detected hand

        if 'lmList' in hand and 'bbox' in hand:  # Ensure hand contains landmarks and bbox
            lmList = hand['lmList']  # Extract landmarks
            x, y, w, h = hand['bbox']  # Extract bounding box

            # Use landmarks to calculate distance between wrist (landmark 5) and pinky base (landmark 17)
            x1, y1 = lmList[5]  # Landmark 5 is the wrist
            x2, y2 = lmList[17]  # Landmark 17 is the pinky base

            # Calculate the Euclidean distance between the two points
            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C  # Convert raw distance to cm

            # Draw bounding box and display distance in cm
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))

    else:
        # Debugging: If no hands are detected
        print("No hands detected!")

    cv2.imshow("Image", img)

    # Quit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
