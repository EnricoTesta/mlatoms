from yaml import safe_load
from utils.docker_utils import get_image_uri
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
    data = safe_load(stream)

current_path = os.path.dirname(os.path.abspath(__file__))

successful_builds = []
successful_image_uris = []
failed_builds = []
failed_image_uris = []
for atom in data['ATOMS'].keys():

    # Notification
    print("Building docker container for: %s" % atom)

    # Build shell command
    cmd = ["sudo cp atoms.py {}/train/atoms.py".format(current_path), "cd train",
           "sudo docker build -f " + data['ATOMS'][atom]['docker_file'] + " -t " + get_image_uri('ATOMS', atom) + " ./",
           "sudo rm {}/train/atoms.py".format(current_path)]

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
    cmd = ["sudo cp atoms.py {}/score/atoms.py".format(current_path), "cd score",
           "sudo docker build -f " + data['SCORING'][scorer]['docker_file'] + " -t " +
           get_image_uri('SCORING', scorer) + " ./", "sudo rm {}/score/atoms.py".format(current_path)]

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

for encoder in data["PREPROCESS"].keys():

    # Notification
    print("Building docker container for: %s" % encoder)

    # Build shell command
    cmd = ["sudo cp atoms.py {}/preprocess/atoms.py".format(current_path), "cd preprocess",
           "sudo docker build -f " + data['PREPROCESS'][encoder]['docker_file'] + " -t " +
           get_image_uri('PREPROCESS', encoder) + " ./", "sudo rm {}/preprocess/atoms.py".format(current_path)]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(encoder)
        successful_image_uris.append(get_image_uri('PREPROCESS', encoder))
    except CalledProcessError as e:
        print("Failed to execute subprocess for container %s. Return code is %s." % (encoder, e.returncode))
        failed_builds.append(encoder)

    # TODO: Test atom locally on sample data
    pass

for encoder in data["POSTPROCESS"].keys():

    # Notification
    print("Building docker container for: %s" % encoder)

    # Build shell command
    cmd = ["sudo cp atoms.py {}/postprocess/atoms.py".format(current_path), "cd postprocess",
           "sudo docker build -f " + data['POSTPROCESS'][encoder]['docker_file'] + " -t " +
           get_image_uri('POSTPROCESS', encoder) + " ./", "sudo rm {}/postprocess/atoms.py".format(current_path)]

    # Execute command
    try:
        check_call(' && '.join(cmd), shell=True)
        successful_builds.append(encoder)
        successful_image_uris.append(get_image_uri('POSTPROCESS', encoder))
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
