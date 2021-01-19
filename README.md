# face_detection
A face detector implementing [S3FD](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf) \[1\] with weights trained on the [WIDER](http://shuoyang1213.me/WIDERFACE/) \[2\] dataset. Implementation of the algorithm is largely based on the code in [this repository](https://github.com/cs-giung/face-detection-pytorch).

## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [OpenCV](https://opencv.org/) (only needed by the test script): `$pip3 install opencv-python`

## How to Install
```
git clone https://github.com/IntelligentBehaviourUnderstandingGroup/face_detection.git
cd face_detection
pip install -e .
```

## How to Test
__Note__: You will need [OpenCV](https://opencv.org/) to run the test script.
* To test on live video: `python face_detection_test.py [-i webcam_index]`
* To test on a video file: `python face_alignment_test.py [-i input_file] [-o output_file]`

## How to Use
```python
from ibug.face_detection import S3FD
# Create a S3FD face detector with the confidence threshold set to 0.8
face_detector = S3FD(thershold=0.8, device='cuda:0')

# Detect faces from an image with (OpenCV's default) B-G-R channel order
# Note:
#   1. The input image must be a byte array of dimension HxWx3.
#   2. The return value is a Nx5 matrix, with each row containing, in this
#      order, the left, top, right, and bottom coordinates, and the confidence 
#      of a detected face
detected_faces = face_detector(image, rgb=False)
```

## References
\[1\] Zhang, Shifeng, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, and Stan Z. Li. "[S3fd: Single shot scale-invariant face detector.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)" In _Proceedings of the IEEE international conference on computer vision_, pp. 192-201. 2017.

\[2\] Yang, Shuo, Ping Luo, Chen-Change Loy, Xiaoou Tang. "[WIDER FACE: A Face Detection Benchmark.](http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf)" In _Proceedings of the IEEE international conference on computer vision_, pp. 5525-5533. 2016.
