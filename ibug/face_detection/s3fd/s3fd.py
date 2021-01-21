import os
import torch
import numpy as np
from types import SimpleNamespace
from .s3fd_net import S3FDNet


class S3FD(object):
    def __init__(self, threshold=0.8, device='cuda:0', weights=None, config=None):
        if weights is None:
            weights = S3FD.get_weights()
        if config is None:
            self.config = S3FD.create_config()
        else:
            self.config = config
        self.device = device
        self.net = S3FDNet(config=self.config, device=self.device).to(self.device)
        state_dict = torch.load(weights, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.threshold = threshold

    @staticmethod
    def get_weights(name='s3fd'):
        name = name.lower()
        if name == 's3fd':
            return os.path.realpath(os.path.join(os.path.dirname(__file__), 'models', 's3fd_weights.pth'))
        else:
            raise ValueError('name must be set to s3fd')

    @staticmethod
    def create_config(num_classes=2, top_k=750, conf_thresh=0.05, variance=(0.1, 0.2),
                      nms_thresh=0.3, nms_top_k=5000, use_nms_np=True,
                      prior_min_sizes=(16, 32, 64, 128, 256, 512),
                      prior_steps=(4, 8, 16, 32, 64, 128), prior_clip=False):
        return SimpleNamespace(num_classes=num_classes, top_k=top_k, conf_thresh=conf_thresh, variance=variance,
                               nms_thresh=nms_thresh, nms_top_k=nms_top_k, use_nms_np=use_nms_np,
                               prior_min_sizes=prior_min_sizes, prior_steps=prior_steps, prior_clip=prior_clip)

    def __call__(self, image, rgb=True):
        w, h = image.shape[1], image.shape[0]
        if not rgb:
            image = image[..., ::-1]
        image = image.astype(int) - np.array([123, 117, 104])
        image = image.transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        image = torch.from_numpy(image).float().to(self.device)

        bboxes = []
        with torch.no_grad():
            detections = self.net(image).detach()
            scale = torch.Tensor([w, h, w, h]).to(detections.device)
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] > self.threshold:
                    score = detections[0, i, j, 0]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    bbox = (pt[0], pt[1], pt[2], pt[3], score)
                    bboxes.append(bbox)
                    j += 1
        if len(bboxes) > 0:
            return np.array(bboxes)
        else:
            return np.empty(shape=(0, 5), dtype=np.float32)
