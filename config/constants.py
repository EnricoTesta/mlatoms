from yaml import safe_load
import os

PATH = os.path.abspath(os.path.dirname(__file__))

# Import global GCP deployment settings from main configuration file
with open("/gauth/y_account_deployment.yml", 'r') as f:
    GLOBALS = safe_load(f)
