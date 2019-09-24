from yaml import safe_load
import os


def get_image_uri(group, name, deployment="GCP"):

    # Read configuration files
    with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
        config = safe_load(stream)
    with open(os.getcwd() + "/config/deploy_config.yml", 'r') as stream:
        deploy_config = safe_load(stream)

    # Fetch URI
    project_id = deploy_config[deployment]['PROJECT_ID']
    image_name = config[group][name]['image_name']
    image_tag = config[group][name]['image_tag']
    uri = deploy_config[deployment]['CONTAINER_ROOT_URL'] + "/" + project_id + "/" + image_name + ":" + image_tag

    return uri
