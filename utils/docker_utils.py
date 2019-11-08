from config.constants import GLOBALS
from yaml import safe_load
import os


def get_image_uri(group, name):

    # Read configuration files
    with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
        config = safe_load(stream)

    # Fetch URI
    project_id = GLOBALS['PROJECT_ID']
    image_name = config[group][name]['image_name']
    image_tag = config[group][name]['image_tag']
    uri = GLOBALS['CONTAINERS_ROOT_URL'] + "/" + project_id + "/" + image_name + ":" + image_tag

    return uri
