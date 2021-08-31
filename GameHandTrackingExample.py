import cv2
import mediapipe as pipe
import time
import HandTrackingModule as htm

cTime = 0
pTime = 0
capture = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
     success, img = capture.read()

     img = cv2.flip(img, 1)
     img = detector.findHands(img)
     
     lmList = detector.finPosition(img,draw=False)
     
     if len(lmList) != 0:
         print(lmList[0])
     
     cTime = time.time()
     fps = 1/(cTime-pTime)
     pTime = cTime
     
     cv2.putText(img, str((int(fps))), (10, 50),
                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)
     
     cv2.imshow("Image", img)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

capture.release()
cv2.destroyAllWindows()
