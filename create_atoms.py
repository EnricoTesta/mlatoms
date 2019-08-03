from yaml import safe_load
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/config.yml", 'r') as stream:
    data = safe_load(stream)

successful_builds = []
failed_builds = []
for key in data['ATOMS'].keys():

    # Notification
    print("Building docker container for: %s" % key)

    # Build shell command
    cmd = ["cd atoms",
           "export PROJECT_ID=$(gcloud config list project --format \"value(core.project)\")",
           "export IMAGE_REPO_NAME=" + data['ATOMS'][key]['image_name'],
           "export IMAGE_TAG=" + data['ATOMS'][key]['image_tag'],
           "export IMAGE_URI=" + data['CONTAINER_ROOT_URL'] + "$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG",
           "sudo docker build -f " + data['ATOMS'][key]['docker'] + " -t $IMAGE_URI ./"]

    # Execute command
    try:
        print("")
        print(' && '.join(cmd))
        print("")
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(key)
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (key, e.returncode))
        failed_builds.append(key)

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
