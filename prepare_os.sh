#!/bin/bash

# Mount volume
cd /dev
FOUND=0
for x in *; do
  if [ $x == "nvme1n1" ]; then
    ${FOUND}=1
    break
  fi
done
if [ ${FOUND} -ne 1 ]; then
  exit 1
fi
sudo mkdir /data
sudo mount /dev/nvme1n1 /data
sudo cp /etc/fstab /etc/fstab.orig
echo "UUID=$(lsblk -nr -o UUID,NAME | grep -Po '.*(?= nvme1n1)')  /data  xfs  defaults,nofail  0  2" >> /etc/fstab
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
