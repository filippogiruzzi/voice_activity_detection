ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SCRIPTS := $(ROOT)scripts

# Clean
clean-docker-images:
	$(eval dangling := $(shell docker images -f dangling=true -q))
	-docker rmi $(dangling)

clean-docker-containers:
	$(eval exited_containers := $(shell docker ps -aq -f status=exited))
	-@[ -z "$(exited_containers)" ] || docker rm $(exited_containers)

clean-docker: clean-docker-containers clean-docker-images

clean-py:
	find . -type d -name "__pycache__" | xargs sudo rm -vrf --

clean: clean-docker clean-py

# Code formatting
check-code:
	$(SCRIPTS)/check_code.sh

# Build
build:
	$(SCRIPTS)/docker_build.sh

# Run
local:
	$(SCRIPTS)/docker_local.sh

# For tests & CI
check-test-docker:
	$(SCRIPTS)/check_tests_docker.sh
