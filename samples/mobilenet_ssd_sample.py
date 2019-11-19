"""
SSD detection sample
 
Command line using example

python mobilenet_ssd_sample.py -p ../models/mobilenet-ssd.xml -m ../models/mobilenet-ssd.bin -i ../data/dog.jpg -t detection -l D:\Intel\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll
"""

import sys
import cv2
import argparse

sys.path.append('../src')
from dnn_detector import DnnDetector
from openvino_dnn_detector import OpenvinoDnnDetector

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
    parser.add_argument("-l", "--cpu_extension",
        help="Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.",
        type=str, default=None)
    return parser

def main():
    print('Hello detection!')
    args = build_argparse().parse_args()
    
    image_src = cv2.imread(args.image)

    detector = OpenvinoDnnDetector(args.model, args.proto, args.task_type,
                                   args.cpu_extension)

    image_detected = detector.detect(image_src)

    cv2.imshow('Image with detections', image_detected)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

if __name__ == '__main__':
    sys.exit(main()) 