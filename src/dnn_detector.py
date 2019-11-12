import cv2
import numpy

class DnnDetector:
    def __init__(self, weightsPath=None, configPath=None,
                 task_type=None, shape = [300,300], scale = 1.0, 
                 mean = [104.0,117.0,123.0]):
        self.weights = weightsPath
        self.config = configPath
        self.task_type = task_type
        self.shape = shape
        self.scale = scale
        self.mean = mean
        # Create net
        self.net = cv2.dnn.readNet(self.weights, self.config)

    def _output_detection(self, output, img):
        (h, w) = img.shape[:2]
        for i in range(0, output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence > 0.001:
                box = output[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
        return img

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, self.scale, self.shape, self.mean)
        self.net.setInput(blob)
        output = self.net.forward()
        return self._output_detection(output, image)