#!/bin/bash

docker run --rm \
		--gpus all \
		--user root \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v /home/filippo/Datasets/perso:/data \
		-v /root:/root \
		-it \
		--entrypoint /bin/bash \
		-e TF_FORCE_GPU_ALLOW_GROWTH=true \
		vad
