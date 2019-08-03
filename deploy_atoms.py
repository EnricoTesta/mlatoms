from yaml import safe_load
from utils.docker_utils import get_image_uri
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/config.yml", 'r') as stream:
    data = safe_load(stream)

successful_pushes = []
failed_pushes = []
for atom in data['ATOMS'].keys():

    # Notification
    print("Pushing docker container for: %s" % atom)

    # Build shell command
    cmd = ["cd atoms",
           "sudo push " + get_image_uri(atom)]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_pushes.append(atom)
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (atom, e.returncode))
        failed_pushes.append(atom)

    # TODO: Test atom locally on sample data
    pass

# Summary
print("")
print("")
print("--- PUSH SUMMARY ---")
for item in successful_pushes:
    print(item + " - success")
for item in failed_pushes:
    print(item + " - failed")
