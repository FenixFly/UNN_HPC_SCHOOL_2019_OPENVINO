import cv2

class Filter():
    def process_image(self):
        pass
           
class GrayScaleFilter(Filter):
    def process_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
        
class ResizeFilter(Filter):
    def process_image(self, image, newX, newY):
        return cv2.resize(image,(int(newX),int(newY)))
        
class SquareCropFilter(Filter):
    def process_image(self, image):
        edge_len = min(image.shape[0:2])
        center = (image.shape[0] // 2, image.shape[1] // 2)
        result = image[center[0] - edge_len // 2 : center[0] + edge_len // 2, 
                        center[1] - edge_len // 2 : center[1] + edge_len // 2]
        return result