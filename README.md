**README**

mlatoms is a docker-based library of Machine Learining algorithms and utilities.
Once you fork the library, all you need to do is run *python create_atoms.py*.
The script will build one docker image for each preprocess, ML, and scoring
algorithm in the library. Once docker images are built you are free to deploy those 
in your favorite environment.

You may run *python deploy_atoms.py* to deploy atoms to GCP Container Registry.

- Make a universal training routine
- Write one script for each algorithm (train)
- Write one script to encapsule each algorithm in an appropriate
docker file (_mlatom_) that can be used to submit jobs on AI platform

These codes will probably be subject to medium evolution depending on the
evolution of the open source libraries they depend on.

These codes should also include openly available autoML libraries in 
order to assess the proprietary algorithms with existing solutions.

Structure
- cloudrun: define GCP Cloud-run ready docker scoring file (WIP)
- config
    - atoms.yml: defines the list of algorithms to build (train, score, preprocess)
    - constants.py: loads GCP global variables (used to generate docker image URIs)
- data: contains a series of toy datasets. Mainly used for local debugging purposes
- preprocess: list of preprocess docker files and entrypoints. 
Preprocessor.py defines universal preprocessing classes
- score: list of score docker files and entrypoints. 
Scorer.py defines universal scoring classes
- train: list of train docker files and entrypoints. Trainer.py defines universal
training behavior.
- utils: docker and GCP utils used across the library. GCP is referenced to build 
docker-image URI
- atoms.py: defines Atom class that implements general behavior inherited by all atom
types (train, score, preprocess)
- create_atoms.py: main script that loops over atom list and builds atoms. Provides a 
build summary.
- deploy_atoms.py: pushes built atoms to GCP Container Registry (WIP)