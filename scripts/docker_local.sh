#!/bin/bash

docker run --rm \
		--gpus all \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-it \
		--entrypoint /bin/bash \
		-e TF_FORCE_GPU_ALLOW_GROWTH=true \
		vad
