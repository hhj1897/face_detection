import os
import subprocess
import sys
import shutil
from setuptools import find_namespace_packages, setup


def clean_repo():
    repo_folder = os.path.realpath(os.path.dirname(__file__))
    dist_folder = os.path.join(repo_folder, 'dist')
    build_folder = os.path.join(repo_folder, 'build')
    if os.path.isdir(dist_folder):
        shutil.rmtree(dist_folder, ignore_errors=True)
    if os.path.isdir(build_folder):
        shutil.rmtree(build_folder, ignore_errors=True)

def pull_first():
    """This script is in a git directory that can be pulled."""
    cwd = os.getcwd()
    gitdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(gitdir)
    try:
        subprocess.call(['git', 'lfs', 'pull'])
    except subprocess.CalledProcessError:
        raise RuntimeError("Make sure git-lfs is installed!")
    os.chdir(cwd)

pull_first()

# Read version string
_version = None
script_folder = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(script_folder, 'ibug', 'face_detection', '__init__.py')) as init:
    for line in init.read().splitlines():
        fields = line.replace('=', ' ').replace('\'', ' ').replace('\"', ' ').replace('\t', ' ').split()
        if len(fields) >= 2 and fields[0] == '__version__':
            _version = fields[1]
            break
if _version is None:
    sys.exit('Sorry, cannot find version information.')

# Installation
config = {
    'name': 'ibug_face_detection',
    'version': _version,
    'description': 'A collection of pretrained face detectors including S3FD and RetinaFace.',
    'author': 'Jie Shen',
    'author_email': 'js1907@imperial.ac.uk',
    'packages': find_namespace_packages(),
    'package_data': {
        'ibug.face_detection.s3fd.weights': ['*.pth'],
        'ibug.face_detection.retina_face.weights': ['*.pth'],
        'ibug.face_detection.utils.data': ['*.npy'],
    },
    'install_requires': ['numpy>=1.16.0', 'scipy>=1.1.0', 'torch>=1.1.0',
                         'torchvision>=0.3.0', 'opencv-python>= 3.4.2'],
    'zip_safe': False
}
clean_repo()
setup(**config)
clean_repo()
