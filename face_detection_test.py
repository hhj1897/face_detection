import os
import cv2
import time
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor, S3FDPredictor


def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
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
    parser.add_argument('--device', '-d', help='Device to be used by the model (default=cuda:0)',
                        default='cuda:0')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the face detector
        args.method = args.method.lower()
        if args.method == 'retinaface':
            face_detector = RetinaFacePredictor(threshold=args.threshold, device=args.device,
                                                model=(RetinaFacePredictor.get_model(args.weights)
                                                       if args.weights else None))
            print('Face detector created using RetinaFace.')
        elif args.method == 's3fd':
            face_detector = S3FDPredictor(threshold=args.threshold, device=args.device,
                                          model=(S3FDPredictor.get_model(args.weights)
                                                 if args.weights else None))
            print('Face detector created using S3FD.')
        else:
            raise ValueError('method must be set to either RetinaFace or S3FD')

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
            out_vid = cv2.VideoWriter(args.output, apiPreference=cv2.CAP_FFMPEG, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        print('Processing started, press \'Q\' to quit.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                elapsed_time = time.time() - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} ms: ' +
                      f'{len(faces)} faces detected.')

                # Rendering
                for face in faces:
                    bbox = face[:4].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
                    if len(face) > 5:
                        for pts in face[5:].reshape((-1, 2)):
                            cv2.circle(frame, tuple(pts.astype(int).tolist()), 3, (0, 0, 255), -1)

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
