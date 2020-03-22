#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Arguments are required"
  exit 1
fi
DRIVE_NAME="$1"
# Mount volume
cd /dev
FOUND=0
for x in *; do
  if [ $x == ${DRIVE_NAME} ]; then
    ${FOUND}=1
    break
  fi
done
if [ ${FOUND} -ne 1 ]; then
  exit 1
fi
sudo mkdir /data
sudo mount /dev/${DRIVE_NAME} /data
sudo cp /etc/fstab /etc/fstab.orig
echo "UUID=$(lsblk -nr -o UUID,NAME | grep -Po '.*(?= ${DRIVE_NAME})')  /data  xfs  defaults,nofail  0  2" | sudo tee -a /etc/fstab
sudo umount /data
sudo mount -a
sudo chown -R $USER /data

sudo apt install x11-xserver-utils

# Install docker
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo groupadd docker
sudo usermod -aG docker ${USER}

#Install CUDA
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install screen to be able to reconnect to same ssh session after connection reset (use 'screen -d -r' after logging in with ssh to re-atach to previous terminal after connection reset)
sudo apt-get install screen
