#!/bin/bash

case "$1" in
    docker)
        docker-compose up -d
        ;;
    pip)
        pip install .
        ;;
esac