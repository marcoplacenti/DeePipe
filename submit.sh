#!/bin/bash

docker build -t deepipe .

docker run -rm deepipe --config config/config.yml 