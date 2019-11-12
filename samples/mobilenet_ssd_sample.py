"""
SSD detection sample
 
Command line using example
 
python mobilenet_ssd_sample.py -p ../models/mobilenet-ssd.prototxt -m ../models/mobilenet-ssd.caffemodel -i ../data/dog.jpg -t detection -me 127.5 127.5 127.5 -s 0.0039 -sh 224 224
"""

import sys
import cv2
import argparse

sys.path.append('../src')
from dnn_detector import DnnDetector

def build_argparse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-p', '--proto', help='Path to an .prototxt \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-m', '--model', help='Path to an .caffemodel file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--image', help='Input image',
        default='', type=str)
    parser.add_argument('-t', '--task_type', help='Task type: \
        detection', default = 'detection', type=str)
    parser.add_argument('-me', '--mean', help='Input mean values', 
                        default = '0 0 0', type=float, nargs=3)
    parser.add_argument('-s', '--scale', help='scale value', 
        required=True, type=float)
    parser.add_argument('-sh', '--shape', help='Network input size',
                        default = '300 300', type=int, nargs=2)
    return parser

def main():
    print('Hello detection!')
    args = build_argparse().parse_args()
    
    image_src = cv2.imread(args.image)

    detector = DnnDetector(args.model, args.proto, args.task_type, tuple(args.shape),
                           args.scale, args.mean)

    image_detected = detector.detect(image_src)

    cv2.imshow('Image with detections', image_detected)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

if __name__ == '__main__':
    sys.exit(main()) 