import time
import cv2
import HandTrackingModule as htm
import pyautogui
import mediapipe as pipe
import numpy as np

capture = cv2.VideoCapture(0)

wCam = 640
hCam = 480

frameRed = 150
smoothen = 7
wScreen, hScreen = pyautogui.size()

cTime = 0
pTime = 0

detector = htm.handDetector(maxHands=1)

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, box = detector.finPosition(img)

    cv2.rectangle(img, (frameRed, frameRed),
                  (wCam-frameRed, hCam-frameRed), (255, 0, 255), 3)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameRed, wCam-frameRed), (0, wScreen))
            y3 = np.interp(y1, (frameRed, hCam-frameRed), (0, hScreen))

            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)

            try:
                pyautogui.moveTo(x3, y3)
            except:
                cv2.putText(img, 'Bring finger inside rectangle', (wCam/2, hCam/2),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255), 3)

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 30:
                pyautogui.click(x3,y3)


    detector.drawFPS(img)
    cv2.imshow("Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
