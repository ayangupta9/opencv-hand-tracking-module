import cv2
import mediapipe as pipe
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = pipe.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = pipe.solutions.drawing_utils
        self.lmList = []
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Drawing connections between landmarks
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

                # get finger tip positions using landmarks

        return img

    def finPosition(self, img, handNo=0, draw=True):

        xList = []
        yList = []
        bbox = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(w*lm.x), int(h * lm.y)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                # if draw:
                #     cv2.circle(img, (cx, cy), 10,
                #                (255, 255, 0), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)

            bbox = [xmin, ymin, xmax, ymax]

            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                              (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    cTime = 0
    pTime = 0
    capture = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = capture.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.finPosition(img)
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


if __name__ == '__main__':
    main()
