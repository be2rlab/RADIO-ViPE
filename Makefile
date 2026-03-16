



# reconstruct-trajectory-from-video:
# 	... VIDEO_PATH TRAJECTORY_PATH


# ------------------------------------------------------------------------------
#                                ALIASES
# ------------------------------------------------------------------------------

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)

# ------------------------------------------------------------------------------

DOCKER_COMPOSE_FILES := \
	-f $(ROOT_DIR)/docker-compose.yml

# ------------------------------------------------------------------------------

RENDER_DISPLAY := $(DISPLAY)
CACHE_DIR?=/home/sber_pn/data/Cache
DATA_DIR?=/home/sber_pn/data
USER_ID=1000#$(UID)
GROUP_ID=1000#$(GID)
DOCKERFILE := Dockerfile

BASE_PARAMETERS := \
	ROOT_DIR=$(ROOT_DIR) \
	CACHE_DIR=$(CACHE_DIR) \
	DATA_DIR=$(DATA_DIR)

# ------------------------------------------------------------------------------

BUILD_COMMAND := DOCKERFILE=$(ROOT_DIR)/docker/$(DOCKERFILE) ROOT_DIR=$(ROOT_DIR) docker compose $(DOCKER_COMPOSE_FILES) build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

RUN_COMMAND := ROOT_DIR=$(ROOT_DIR) docker compose $(DOCKER_COMPOSE_FILES) run --rm


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                              BUILDING COMMANDS
# ------------------------------------------------------------------------------

build:
	@echo "Building"
	cd $(ROOT_DIR) && $(BUILD_COMMAND) vipe


# ------------------------------------------------------------------------------
#                              RUNNING COMMANDS
# ------------------------------------------------------------------------------

run:
	@echo "Running"
	cd $(ROOT_DIR) && \
	export $(BASE_PARAMETERS) && \
	$(RUN_COMMAND) vipe

# ------------------------------------------------------------------------------
#                             AUXILIARY COMMANDS
# ------------------------------------------------------------------------------

prepare-terminal-for-visualization:
	xhost +local:
	DISPLAY=$(DISPLAY) xhost +
	RCUTILS_COLORIZED_OUTPUT=1
