from yaml import safe_load
from subprocess import check_call, CalledProcessError
import os

# Read configuration file
with open(os.getcwd() + "/config/atoms.yml", 'r') as stream:
    data = safe_load(stream)

with open(os.getcwd() + "/config/deploy_config.yml", 'r') as stream:
    DEPLOYMENT_GLOBALS = safe_load(stream)

successful_builds = []
failed_builds = []
for atom in data['ATOMS'].keys():

    # Notification
    print("Building prediction app for: %s" % atom)

    prediction_sources = data['ATOMS'][atom]['prediction_sources']

    # Build shell command
    cmd = ["cd " + prediction_sources,
           "python setup.py sdist"]

    # Execute command
    # TODO: check why the distribution command fails programmatically (setuptools not found) but works manually.
    try:
        check_call(' && '.join(cmd), shell=True)
    except CalledProcessError as e:
        print("Failed to build prediction app for container %s. Return code is %s." % (atom, e.returncode))
        failed_builds.append(atom)
        continue

    # Copy app to GCS
    ref_folder = prediction_sources + "dist/"
    distro = [f for f in os.listdir(ref_folder)
              if os.path.isfile(os.path.join(ref_folder, f))][0]  # assume exactly 1 distro
    distro_full_path = ref_folder + distro
    cmd_gcs = "gsutil cp " + distro_full_path + " gs://" + DEPLOYMENT_GLOBALS["GCP"] + "/"
    try:
        check_call(cmd_gcs, shell=True)
        successful_builds.append(atom)
    except CalledProcessError as e:
        print("Failed to build deploy prediction app to GCS for container %s. Return code is %s." % (atom, e.returncode))
        failed_builds.append(atom)

# Summary
print("")
print("")
print("--- PREDICTION APP BUILD SUMMARY ---")
for item in successful_builds:
    print(item + " - success")
for item in failed_builds:
    print(item + " - failed")
