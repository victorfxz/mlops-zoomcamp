#!/bin/bash

docker-compose up -d

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

python integration_test.py

aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/out/

# Parar Localstack
docker-compose down