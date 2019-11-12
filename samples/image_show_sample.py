import sys
import cv2

def main():
    print('Hello OpenCV! It is showing image sample')
     
    image_src = cv2.imread('../data/logo.png')
    cv2.imshow('Show image sample', image_src)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    
if __name__ == '__main__':
    sys.exit(main())