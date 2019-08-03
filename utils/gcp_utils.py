from yaml import safe_load
import os


def get_container_registry_service_account_json():

    # Read configuration file
    with open(os.getcwd() + "/config/deploy_config.yml", 'r') as stream:
        config = safe_load(stream)

    return config["GCP"]["CONTAINER_REGISTRY_SERVICE_ACCOUNT_JSON"]


def get_registry_hostname():

    # Read configuration file
    with open(os.getcwd() + "/config/deploy_config.yml", 'r') as stream:
        deploy_config = safe_load(stream)

    return deploy_config["GCP"]['CONTAINER_ROOT_URL']