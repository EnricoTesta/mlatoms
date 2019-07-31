from setuptools import setup
from setuptools import find_packages

#REQUIRED_PACKAGES = ['scikit-learn==0.20.2']

setup(
    name='my_custom_predictor',
    #install_requires=REQUIRED_PACKAGES,
    #packages=find_packages(),
    #include_package_data=True,
    version='0.1',
    scripts=['predictor.py', 'preprocess.py'])
