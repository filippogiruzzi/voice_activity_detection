#!/bin/bash

ROOT=$1

docker run --rm \
		--gpus all \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v $ROOT:/voice_activity_detection \
		-it \
		--entrypoint /bin/bash \
		-e TF_FORCE_GPU_ALLOW_GROWTH=true \
		vad
