#!/bin/bash

echo "Installing Docker"

sudo apt-get -y install ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get -y update

# pick the latest patch from:
# apt-cache madison docker-ce | awk '{ print $3 }' | grep -i 20.10
VERSION_STRING=5:20.10.24~3-0~ubuntu-jammy
sudo apt-get install docker-ce-cli=$VERSION_STRING docker-compose-plugin -y

# validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]
docker version

echo "Installing Local SageMaker Tarball"
pip install "pydantic>=2.0.0"
pip install sagemaker-2.232.4.dev0.tar.gz


echo "Setting Up Read-Only SSH Access"
eval "$(ssh-agent -s)"

mkdir -p ~/.ssh/

cp /home/sagemaker-user/bootstrap/adapter_deploy_key /home/sagemaker-user/.ssh/adapter_deploy_key
chmod 600 ~/.ssh/adapter_deploy_key

cp /home/sagemaker-user/bootstrap/launcher_deploy_key /home/sagemaker-user/.ssh/launcher_deploy_key 
chmod 600 ~/.ssh/launcher_deploy_key

cp /home/sagemaker-user/bootstrap/config /home/sagemaker-user/.ssh/config
chmod 644 ~/.ssh/config

ssh-add ~/.ssh/adapter_deploy_key
ssh-add ~/.ssh/launcher_deploy_key



