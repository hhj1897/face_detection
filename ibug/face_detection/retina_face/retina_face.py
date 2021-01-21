import os


class RetinaFace(object):
    def __init__(self, threshold=0.8, device='cuda:0', weights=None):
        pass

    @staticmethod
    def get_weights(name='resnet50'):
        name = name.lower()
        if name == 'resnet50':
            return os.path.realpath(os.path.join(os.path.dirname(__file__), 'models', 'Resnet50_Final.pth'))
        elif name == 'mobilenet0.25':
            return os.path.realpath(os.path.join(os.path.dirname(__file__), 'models', 'mobilenet0.25_Final.pth'))
        else:
            raise ValueError('name must be set to either resnet50 or mobilenet0.25')

    def __call__(self):
        pass
