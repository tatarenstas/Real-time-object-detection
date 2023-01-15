import cv2
import time
import numpy as np

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList = np.random.uniform(low = 0, high = 255, size = (len(self.classesList), 3))

        #print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Error opening file")
            return

        (success, image) = cap.read()

        start_time = 0

        while success:
            current_time = time.time()
            fps = 1/(current_time - start_time)
            start_time = current_time

            scale_percent = 50
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            classLabelIDs, confindences, bboxs = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confindences = list(np.array(confindences).reshape(1, -1)[0])
            confindences = list(map(float, confindences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confindences, score_threshold = 0.5, nms_threshold = 0.2)
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confindences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{} : {:.2f}".format(classLabel, classConfidence)
                    #print(displayText)

                    x,y,w,h = bbox

                    cv2.rectangle(image, (x,y), (x+w, y+h), color = classColor, thickness = 1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", image)

            if cv2.waitKey(1) % 0xFF == ord('q'):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()