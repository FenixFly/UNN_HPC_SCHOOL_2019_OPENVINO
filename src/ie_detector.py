import cv2
import numpy
from openvino.inference_engine import IENetwork, IECore

class InferenceEngineDetector:
    def __init__(self, weightsPath=None, configPath=None,
                 device='CPU', extension = None):
        #
        # Add your code here
        #
        
        return

    def _output_detection(self, output, img):
    
        #
        # Add your code here
        #
        
        return img

    def _prepare_image(self, image, h, w):
    
        #
        # Add your code here
        #
        
        return image
        
    def detect(self, image):
    
        #
        # Add your code here
        #
        
        return self._output_detection(output, image)