from yaml import safe_load
import os


def get_image_uri(atom_name):

    # Read configuration file
    with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
        config = safe_load(stream)

    # Fetch URI
    project_id = config['PROJECT_ID']
    image_name = config['ATOMS'][atom_name]['image_name']
    image_tag = config['ATOMS'][atom_name]['image_tag']
    uri = config['CONTAINER_ROOT_URL'] + project_id + "/" + image_name + ":" + image_tag

    return uri
