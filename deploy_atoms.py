from yaml import safe_load
from utils.docker_utils import get_image_uri
from utils.gcp_utils import get_container_registry_service_account_json, get_registry_hostname
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
    data = safe_load(stream)

# Get permission to push to registry
try:
    # This was successful when sent manually
    "cat ~/mlatoms/deploy/container_registry_ref_file.json | docker login -u _json_key --password-stdin https://gcr.io"

    # If you try to do it again it says
    "docker login requires at most 1 argument"


    cmd = "cat " + get_container_registry_service_account_json() + \
          " | docker login -u _json_key --password-stdin https://" + get_registry_hostname()

    cmd_2 = 'docker login -u _json_key -p /"$(cat ' + get_container_registry_service_account_json() + \
            ')/" https://' + get_registry_hostname()

    print("")
    print(cmd_2)
    print("")
    check_call(cmd_2)
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
        check_call(cmd, shell=True)
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
