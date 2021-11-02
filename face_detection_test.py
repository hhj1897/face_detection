import os
import cv2
import time
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor, S3FDPredictor
from ibug.face_detection.utils import SimpleFaceTracker, HeadPoseEstimator


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
    parser.add_argument('--fourcc', '-f', help='FourCC of the output video (default=mp4v)',
                        type=str, default='mp4v')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--no-display', '-n', help='No display if processing a video file',
                        action='store_true', default=False)
    parser.add_argument('--threshold', '-t', help='Confidence threshold (default=0.8)',
                        type=float, default=0.8)
    parser.add_argument('--method', '-m', help='Method to use, can be either RatinaFace or S3FD (default=RatinaFace)',
                        default='retinaface')
    parser.add_argument('--weights', '-w',
                        help='Weights to load, can be either resnet50 or mobilenet0.25 when using RetinaFace',
                        default=None)
    parser.add_argument('--alternative-pth', '-p', help='Alternative pth file to load', default=None)
    parser.add_argument('--device', '-d', help='Device to be used by the model (default=cuda:0)',
                        default='cuda:0')
    parser.add_argument('--iou-threshold', '-iou',
                        help='IOU threshold used by the simple face tracker (default=0.4)',
                        type=float, default=0.4)
    parser.add_argument('--minimum-face-size', '-min',
                        help='Minimum face size used by the simple face tracker (default=0.0)',
                        type=float, default=0.0)
    parser.add_argument('--head-pose-preference', '-hp',
                        help='Head pose output preference (default=0)',
                        type=int, default=0)
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the face detector
        args.method = args.method.lower().strip()
        if args.method == 'retinaface':
            face_detector_class = (RetinaFacePredictor, 'RetinaFace')
        elif args.method == 's3fd':
            face_detector_class = (S3FDPredictor, 'S3FD')
        else:
            raise ValueError('method must be set to either RetinaFace or S3FD')
        if args.weights is None:
            fd_model = face_detector_class[0].get_model()
        else:
            fd_model = face_detector_class[0].get_model(args.weights)
        if args.alternative_pth is not None:
            fd_model.weights = args.alternative_pth
        face_detector = face_detector_class[0](threshold=args.threshold, device=args.device, model=fd_model)
        print(f"Face detector created using {face_detector_class[1]} ({fd_model.weights}).")

        # Create the simple face tracker
        face_tracker = SimpleFaceTracker(iou_threshold=args.iou_threshold,
                                         minimum_face_size=args.minimum_face_size)
        print('Simple face tracker created.')

        # Create the head pose estimator
        head_pose_estimator = HeadPoseEstimator()
        print('Head pose estimator created.')

        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        if using_webcam:
            print(f'Webcam #{int(args.input)} opened.')
        else:
            print(f'Input video "{args.input}" opened.')

        # Open the output video (if a path is given)
        if args.output is not None:
            out_vid = cv2.VideoWriter(args.output, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc(*args.fourcc))
            assert out_vid.isOpened()

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
        print('Processing started, press \'Q\' to quit or \'R\' to reset the tracker.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect and track faces, also estimate head pose if landmarks are available
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                tids = face_tracker(faces)
                if faces.shape[1] >= 15:
                    head_poses = [head_pose_estimator(face[5:15].reshape((-1, 2)), *frame.shape[1::-1],
                                                      output_preference=args.head_pose_preference)
                                  for face in faces]
                else:
                    head_poses = [None] * faces.shape[0]
                elapsed_time = time.time() - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} ms: ' +
                      f'{len(faces)} faces detected.')

                # Rendering
                for face, tid, head_pose in zip(faces, tids, head_poses):
                    bbox = face[:4].astype(int)
                    if tid is None:
                        colour = (128, 128, 128)
                    else:
                        colour = colours[(tid - 1) % len(colours)]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=colour, thickness=2)
                    if len(face) > 5:
                        for pts in face[5:].reshape((-1, 2)):
                            cv2.circle(frame, tuple(pts.astype(int).tolist()), 3, colour, -1)
                    if tid is not None:
                        cv2.putText(frame, f'Face {tid}', (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, lineType=cv2.LINE_AA)
                    if head_pose is not None:
                        pitch, yaw, roll = head_pose
                        cv2.putText(frame, f'Pitch: {pitch:.1f}', (bbox[2] + 5, bbox[1] + 10),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, lineType=cv2.LINE_AA)
                        cv2.putText(frame, f'Yaw: {yaw:.1f}', (bbox[2] + 5, bbox[1] + 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, lineType=cv2.LINE_AA)
                        cv2.putText(frame, f'Roll: {roll:.1f}', (bbox[2] + 5, bbox[1] + 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, lineType=cv2.LINE_AA)

                # Write the frame to output video (if recording)
                if out_vid is not None:
                    out_vid.write(frame)

                # Display the frame
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord('q') or key == ord('Q'):
                        print('\'Q\' pressed, we are done here.')
                        break
                    elif key == ord('r') or key == ord('R'):
                        print('\'R\' pressed, reset the tracker.')
                        face_tracker.reset()
                frame_number += 1
    finally:
        if has_window:
            cv2.destroyAllWindows()
        if out_vid is not None:
            out_vid.release()
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
