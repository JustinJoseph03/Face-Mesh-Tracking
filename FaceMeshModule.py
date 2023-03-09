import cv2
import mediapipe as mp
import time

class FaceMeshDetector():

    def __int__(self, staticMode = False, maxFaces = 2,
                refineLm = False, minDetectCon = 0.5,
                minTrackCon = 0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLm = refineLm
        self.minDetectCon = minDetectCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,
                                                 self.refineLm,self.minDetectCon,
                                                 self.minTrackCon)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color = (0,255,0))

    def findFaceMesh(self, img, draw = True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)          #Not recognizing faceMesh atm
        if self.results.multi_face_landmarks:
            for faceLandmarks in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                      self.drawSpec, self.drawSpec)

                # for id, landmarks in enumerate(faceLandmarks.landmark):
                #     #print (landmarks)
                #     ih,iw,ic = img.shape
                #     x,y, = int(landmarks.x*iw), int(landmarks.y*ih)
                #     print(id,x,y)

        return img

def main():
    cap = cv2.VideoCapture("Videos/1.mp4")
    prevTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)

        currTime = time.time()
        fps = 1 /(currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()