ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SCRIPTS := $(ROOT)scripts


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
	find . -type d -name "__pycache__" | xargs sudo rm -vrf --

clean: clean-docker clean-py

##################################################
#                 Code formatting                #
##################################################
check-code:
	$(SCRIPTS)/check_code.sh

##################################################
#                 Docker commands                #
##################################################
build:
	$(SCRIPTS)/docker_build.sh

local: build
	$(SCRIPTS)/docker_local.sh

local-nobuild:
	$(SCRIPTS)/docker_local.sh

##################################################
#                    CI & tests                  #
##################################################
check-test-docker: build
	$(SCRIPTS)/check_tests_docker.sh
