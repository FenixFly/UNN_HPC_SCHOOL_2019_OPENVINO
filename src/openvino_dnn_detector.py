import cv2
import numpy
from openvino.inference_engine import IENetwork, IECore

class OpenvinoDnnDetector:
    def __init__(self, weightsPath=None, configPath=None,
                 task_type=None, cpu_extension = None):
        self.weights = weightsPath
        self.config = configPath
        self.task_type = task_type
        # Create net
        #self.net = cv2.dnn.readNet(self.weights, self.config)
        self.ie = IECore()
        self.net = IENetwork(model=configPath, weights=weightsPath)
        if cpu_extension:
            self.ie.add_extension(cpu_extension, 'CPU')
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')

    def _output_detection(self, output, img):
        (h, w) = img.shape[:2]
        for i in range(0, output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence > 0.5:
                print(i, confidence)
                box = output[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                print(box)
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
        return img

    def prepare_image(self, image, h, w):
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        return image
        
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
    
        blob = self.prepare_image(image, h, w)
    
        output = self.exec_net.infer(inputs={input_blob: blob})
        output = output[out_blob]
        print(output.shape, output)
        
        return self._output_detection(output, image)