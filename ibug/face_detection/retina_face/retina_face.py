import os
from types import SimpleNamespace


class RetinaFace(object):
    def __init__(self, threshold=0.8, device='cuda:0', weights=None, config=None):
        if weights is None:
            weights = RetinaFace.get_weights()
        if config is None:
            self.config = RetinaFace.create_config()
        else:
            self.config = config

    @staticmethod
    def get_weights(name='resnet50'):
        name = name.lower()
        if name == 'resnet50':
            return os.path.realpath(os.path.join(os.path.dirname(__file__), 'weights', 'Resnet50_Final.pth'))
        elif name == 'mobilenet0.25':
            return os.path.realpath(os.path.join(os.path.dirname(__file__), 'weights', 'mobilenet0.25_Final.pth'))
        else:
            raise ValueError('name must be set to either resnet50 or mobilenet0.25')

    @staticmethod
    def create_config():
        return SimpleNamespace()

    def __call__(self):
        pass
