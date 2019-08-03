from yaml import safe_load
import os


def get_docker_bridge_service_account():

    # Read configuration file
    with open(os.getcwd() + "/config/deploy_config.yml", 'r') as stream:
        config = safe_load(stream)

    return config["GCP"]["DOCKER_BRIDGE_SERVICE_ACCOUNT"]
