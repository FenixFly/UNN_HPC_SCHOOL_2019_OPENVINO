import sys
sys.path.append('../src')

import cv2
from image_filter import GrayScaleFilter, SquareCropFilter, ResizeFilter

def main():
    print('Hello filtering!')
    
    image_src = cv2.imread('../data/logo.png')
    
    grayFilter = GrayScaleFilter()
    image_gray = grayFilter.process_image(image_src) 
    
    crop = SquareCropFilter()
    image_cropped = crop.process_image(image_gray) 
    
    resize = ResizeFilter()
    image_resized = resize.process_image(image_cropped, 256, 256)
    
    cv2.imshow('Gray image', image_resized)
    cv2.waitKey(0) # waits until a key is pressed
    
    cv2.destroyAllWindows() # destroys the window showing image
    
if __name__ == '__main__':
    sys.exit(main())