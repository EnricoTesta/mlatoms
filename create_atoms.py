from yaml import safe_load
from utils.docker_utils import get_image_uri
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
    data = safe_load(stream)

successful_builds = []
successful_image_uris = []
failed_builds = []
failed_image_uris = []
for atom in data['ATOMS'].keys():

    # Notification
    print("Building docker container for: %s" % atom)

    # Build shell command
    cmd = ["cd atoms",
           "sudo docker build -f " + data['ATOMS'][atom]['docker_file'] + " -t " + get_image_uri('ATOMS', atom) + " ./"]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(atom)
        successful_image_uris.append(get_image_uri('ATOMS', atom))
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (atom, e.returncode))
        failed_builds.append(atom)

    # TODO: Test atom locally on sample data
    pass

for scorer in data['SCORING'].keys():

    # Notification
    print("Building docker container for: %s" % scorer)

    # Build shell command
    cmd = ["cd scoring",
           "sudo docker build -f " + data['SCORING'][scorer]['docker_file'] + " -t " + get_image_uri('SCORING', scorer) + " ./"]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(scorer)
        successful_image_uris.append(get_image_uri('SCORING', scorer))
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (scorer, e.returncode))
        failed_builds.append(scorer)

    # TODO: Test atom locally on sample data
    pass

for encoder in data["PREPROCESSING"].keys():

    # Notification
    print("Building docker container for: %s" % encoder)

    # Build shell command
    cmd = ["cd preprocess",
           "sudo docker build -f " + data['PREPROCESSING'][encoder]['docker_file'] + " -t " +
           get_image_uri('PREPROCESSING', encoder) + " ./"]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(encoder)
        successful_image_uris.append(get_image_uri('SCORING', encoder))
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (encoder, e.returncode))
        failed_builds.append(encoder)

    # TODO: Test atom locally on sample data
    pass

# Summary
print("")
print("")
print("--- BUILD SUMMARY ---")
for i, item in enumerate(successful_builds):
    print("{} - {} - success".format(item, successful_image_uris[i]))
for i, item in enumerate(failed_builds):
    print("{} - {} - failed".format(item, successful_image_uris[i]))
