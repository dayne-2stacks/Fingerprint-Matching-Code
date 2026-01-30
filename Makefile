COMPOSE ?= docker compose
.PHONY: build run up down shell logs

build:
	$(COMPOSE) build

run:
	$(COMPOSE) run --rm app

up:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down

shell:
	$(COMPOSE) run --rm app /bin/bash

logs:
	$(COMPOSE) logs -f
