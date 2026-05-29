ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SCRIPTS := $(ROOT)scripts


##################################################
#                    Setup                       #
##################################################
install:
	uv sync

##################################################
#                 Code formatting                #
##################################################
lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

##################################################
#                    Tests                       #
##################################################
test:
	uv run pytest --cov=vad --cov-report=term-missing --cov-report=html

##################################################
#   Clean docker images, containers & pycache    #
##################################################
clean-docker-images:
	$(eval dangling := $(shell docker images -f dangling=true -q))
	-docker rmi $(dangling)

clean-docker-containers:
	$(eval exited_containers := $(shell docker ps -aq -f status=exited))
	-@[ -z "$(exited_containers)" ] || docker rm $(exited_containers)

clean-docker: clean-docker-containers clean-docker-images

clean-py:
	find . -type d -name "__pycache__" | xargs rm -rf

clean: clean-docker clean-py

##################################################
#                 Docker commands                #
##################################################
build:
	$(SCRIPTS)/docker_build.sh

build-gpu:
	$(SCRIPTS)/docker_build.sh --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

local: build
	$(SCRIPTS)/docker_local.sh

local-nobuild:
	$(SCRIPTS)/docker_local.sh

##################################################
#                    CI & tests                  #
##################################################
check-test-docker: build
	$(SCRIPTS)/check_tests_docker.sh
