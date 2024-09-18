from setuptools import setup, find_packages

setup(
    name='crack_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'opencv-python',
        'segmentation-models',
        'matplotlib',  # Added matplotlib
    ],
)