from setuptools import setup

setup(
    name='logreg-predictor',
    version='0.1',
    #py_modules=['templates.predictor'],
    scripts=['predictor.py', 'preprocess.py'])

# TODO: package build looks good, however it fails to install on GCP AI platform
