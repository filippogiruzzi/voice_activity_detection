#!/bin/bash

docker build . -t vad_test -f ./tests/Dockerfile

docker run --rm -t vad_test
