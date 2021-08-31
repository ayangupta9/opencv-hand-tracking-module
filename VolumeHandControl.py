import cv2
import time
import mediapipe as pipe
import HandTrackingModule as htm
import numpy as np
import math
import pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

capture = cv2.VideoCapture(0)

cTime = 0
pTime = 0

detector = htm.handDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, box = detector.finPosition(img, draw=True)

    if len(lmList) != 0:
            area = (box[2] - box[0]) * (box[3] - box[1]) // 100
            if 500 < area < 1000:

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1+x2)//2, (y1+y2)//2

            length = math.dist([x1, y1], [x2, y2])

            cv2.circle(img, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 4)
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

            vol = np.interp(length, [30, 150], [0, 1])
            volBar = np.interp(length, [30, 150], [400, 150])
            volPer = np.interp(length, [30, 150], [0, 100])
            volume.SetMasterVolumeLevelScalar(vol, None)

            if length < 30:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(volPer)), (10, 450),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 3)

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    cv2.putText(img, str(fps), (10, 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)

    cv2.imshow("Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
