from setuptools import setup

setup(
    name='logreg-predictor',
    version='0.1',
    scripts=['predictor.py', 'logregApp/logreg_predictor.py', 'logregApp/preprocess.py'])

# TODO: package build looks good, however it fails to install on GCP AI platform
