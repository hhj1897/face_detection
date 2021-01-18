import os
import torch
import numpy as np
from .s3fd_net import S3FDNet


class S3FD(object):
    def __init__(self, threshold=0.8, device='cuda:0',
                 weights_path=os.path.join(os.path.dirname(__file__), 'models', 's3fd_weights.pth')):
        self.device = device
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.threshold = threshold

    def __call__(self, image, rgb=False):
        w, h = image.shape[1], image.shape[0]
        if rgb:
            image = image[..., ::-1]
        image = image - np.array([104.0, 117.0, 123.0])
        image = image.transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        image = torch.from_numpy(image).float().to(self.device)

        bboxes = []
        with torch.no_grad():
            detections = self.net(image)
            scale = torch.Tensor([w, h, w, h])
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] > self.threshold:
                    score = detections[0, i, j, 0]
                    pt = (detections[0, i, j, 1:] * scale).detach().cpu().numpy()
                    bbox = (pt[0], pt[1], pt[2], pt[3], score)
                    bboxes.append(bbox)
                    j += 1
        if len(bboxes) > 0:
            return np.array(bboxes)
        else:
            return np.empty(shape=(0, 5), dtype=np.float32)
