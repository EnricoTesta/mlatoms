from yaml import safe_load
from utils.docker_utils import get_image_uri
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/config.yml", 'r') as stream:
    data = safe_load(stream)

successful_builds = []
failed_builds = []
for atom in data['ATOMS'].keys():

    # Notification
    print("Building docker container for: %s" % atom)

    # Build shell command
    cmd = ["cd atoms",
           "sudo docker build -f " + data['ATOMS'][atom]['docker_file'] + " -t " + get_image_uri(atom) + " ./"]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(atom)
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (atom, e.returncode))
        failed_builds.append(atom)

    # TODO: Test atom locally on sample data
    pass

# Summary
print("")
print("")
print("--- BUILD SUMMARY ---")
for item in successful_builds:
    print(item + " - success")
for item in failed_builds:
    print(item + " - failed")
