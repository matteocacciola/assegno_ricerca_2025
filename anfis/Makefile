PWD = $(shell pwd)

LOCAL_DIR = $(PWD)/venv/bin
PYTHON = $(LOCAL_DIR)/python
PYTHON3 = python3.10

args=

help:  ## Show help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

sync-requirements: ## Update the local virtual environment as well as the container with the latest requirements.
	$(PYTHON) -m pip install --no-cache-dir -r requirements.txt

run:  ## Run the application
	$(PYTHON) -m main ${args}