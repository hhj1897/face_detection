# ibug.face_detection
A collection of pretrained face detectors including:
* [S3FD](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf) \[1\] with weights trained on the [WIDER](http://shuoyang1213.me/WIDERFACE/) \[2\] dataset. Implementation of the algorithm is based on this repository: [https://github.com/cs-giung/face-detection-pytorch](https://github.com/cs-giung/face-detection-pytorch).
* [RetinaFace](https://arxiv.org/pdf/1905.00641) \[3\] with weights trained on the [WIDER](http://shuoyang1213.me/WIDERFACE/) \[2\] dataset. Wights for networks using either Resnet50 or MobileNet0.25 as the backbone are included. The implementation is based on this repository: [https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface).

For convenience, the package also includes a simple IOU-based face tracker and a head pose estimator using EPnP.

## Prerequisites
* [Git LFS](https://git-lfs.github.com/), needed for downloading the pretrained weights that are larger than 100 MB.
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [Sciypy](https://www.scipy.org/): `$pip3 install scipy`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`

## How to Install
```
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
```

## How to Test
* To test on live video: `python face_detection_test.py [-i webcam_index]`
* To test on a video file: `python face_detection_test.py [-i input_file] [-o output_file]`

By default, the test script would use RetinaFace with Resnet50, but you can change that using `--method` and `--weights` options. 

## How to Use
```python
# Import everything, just for illustration purposes
import cv2
from ibug.face_detection import RetinaFacePredictor, S3FDPredictor
from ibug.face_detection.utils import HeadPoseEstimator, SimpleFaceTracker

# Create a RetinaFace detector using Resnet50 backbone, with the confidence 
# threshold set to 0.8
face_detector = RetinaFacePredictor(
    threshold=0.8, device='cuda:0',
    model=RetinaFacePredictor.get_model('resnet50'))

# Create a head pose estimator
pose_estimator = HeadPoseEstimator()

# Create a simple face tracker, with mininum face size set to 64x64 pixels
face_tracker = SimpleFaceTracker(minimum_face_size=64)

# Load a test image. Note that images loaded by OpenCV adopt the B-G-R channel
# order.
image = cv2.imread('test.png')

# Detect faces from the image
# Note:
#   1. The input image must be a byte array of dimension HxWx3.
#   2. The return value is a Nx5 (for S3FD) or a Nx15 (for RetinaFace) matrix,
#      in which N is the number of detected faces. The first 4 columns store 
#      (in this order) the left, top, right, and bottom coordinates of the 
#      detected face boxes. The 5th columns stores the detection confidences.
#      The remaining columns store the coordinates (in the order of x1, y1, x2,
#      y2, ...) of the detected landmarks.
detected_faces = face_detector(image, rgb=False)

# Head pose estimation (only works for RetinaFace, which also detects the 5
# landmarks on the face), which gives pitch, yaw, and roll (in degrees) of
# the detected faces.
for face in detected_faces:
    pitch, yaw, roll = pose_estimator(face[5:].reshape((-1, 2)))

# If you are processing frames in a video, you can also perform rudimentary
# face tracking, as shown below. The return value is a list containing the 
# tracklet ID (>=1) of the detected faces. If a face cannot be tracked 
# (such as because it is too small), its corresponding element in the list 
# would be set to None.
tracked_ids = face_tracker(detected_faces[:, :4])
```

## References
\[1\] Zhang, Shifeng, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, and Stan Z. Li. "[S3fd: Single shot scale-invariant face detector.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)" In _Proceedings of the IEEE international conference on computer vision_, pp. 192-201. 2017.

\[2\] Yang, Shuo, Ping Luo, Chen-Change Loy, Xiaoou Tang. "[WIDER FACE: A Face Detection Benchmark.](http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf)" In _Proceedings of the IEEE international conference on computer vision_, pp. 5525-5533. 2016.

\[3\] Deng, Jiankang, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. "[Retinaface: Single-shot multi-level face localisation in the wild.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf)" In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 5203-5212. 2020.
