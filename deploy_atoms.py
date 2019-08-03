from yaml import safe_load
from utils.docker_utils import get_image_uri
from utils.gcp_utils import get_docker_bridge_service_account
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
    data = safe_load(stream)

# Get permission to push to registry
try:
    check_call("sudo usermod -a -G docker " + get_docker_bridge_service_account())
except CalledProcessError as e:
    print("Failed to obtain access rights to repository. Subprocess return code is %s" % e.returncode)
    print("Aborting...")
    exit()

successful_pushes = []
failed_pushes = []
for atom in data['ATOMS'].keys():

    # Notification
    print("Pushing docker container for: %s" % atom)

    # Build shell command
    cmd = "sudo docker push " + get_image_uri(atom)

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_pushes.append(atom)
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (atom, e.returncode))
        failed_pushes.append(atom)

# Summary
print("")
print("")
print("--- PUSH SUMMARY ---")
for item in successful_pushes:
    print(item + " - success")
for item in failed_pushes:
    print(item + " - failed")
