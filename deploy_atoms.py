from yaml import safe_load
from utils.docker_utils import get_image_uri
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
    data = safe_load(stream)

successful_pushes = []
failed_pushes = []
successful_image_uris = []
for atom_group in ['ATOMS', 'SCORING', 'PREPROCESS', 'POSTPROCESS']:
    for atom in data[atom_group].keys():

        # Notification
        print("Pushing docker container for: %s" % atom)

        # Build shell command
        cmd = "sudo docker push " + get_image_uri(atom_group, atom)

        # Execute command
        try:
            check_call(cmd, shell=True)
            successful_pushes.append(atom)
            successful_image_uris.append(get_image_uri(atom_group, atom))
        except CalledProcessError as e:
            print("Failed to execute subprocess for container %s. Return code is %s." % (atom, e.returncode))
            failed_pushes.append(atom)

# Summary
print("")
print("")
print("--- PUSH SUMMARY ---")
for i, item in enumerate(successful_pushes):
    print("{} - {} - success".format(item, successful_image_uris[i]))
for i, item in enumerate(failed_pushes):
    print("{} - {} - failed".format(item, successful_image_uris[i]))
