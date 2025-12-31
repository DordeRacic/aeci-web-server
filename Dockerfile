FROM ubuntu:latest
LABEL authors="djord"

ENTRYPOINT ["top", "-b"]