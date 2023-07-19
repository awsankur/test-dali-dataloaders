#!/bin/bash

source docker.properties


# Edit for using multiple chips
docker run -t -d --name ${docker_container_name} -v /home/ubuntu/test-dali-dataloaders/multi-channel-images/data/:/data/ ${registry}/${docker_image_name} 
