import cv2
import mediapipe as pipe
import time
import HandTrackingModule as hm

capture = cv2.VideoCapture(0)
capture.set(3, 720)
capture.set(4, 480)

detector = hm.handDetector()

countValues = [1, 2, 3, 4, 5, 0]
tipIds = [4, 8, 12, 16, 20]
cTime = 0
pTime = 0

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)

    lmList = detector.finPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        cv2.putText(img, str(totalFingers),
                    (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    cv2.putText(img, str(fps), (550, 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)

    cv2.imshow("Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
