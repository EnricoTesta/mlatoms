#!/bin/bash


#Choose Ubuntu 18.10 (GNU/Linux 4.18.0-1005-gcp x86_64)
#add 20 GB disk + allow http and http
#set access for each API -> Storage : Read Write

sudo snap remove google-cloud-sdk

curl https://sdk.cloud.google.com | bash

#reconnect to the VM

install docker https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install \
apt-transport-https \
ca-certificates \
curl \
gnupg-agent \
software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker run hello-world # test

sudo usermod -a -G docker <YOUR_USERNAME>

#reconnect to the VM

gcloud auth configure-docker
