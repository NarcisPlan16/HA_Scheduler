# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

### Base Image Build ###
#FROM ghcr.io/home-assistant/aarch64-base-python:latest
#FROM ghcr.io/home-assistant/aarch64-base-ubuntu:20.04
FROM ghcr.io/home-assistant/amd64-base-ubuntu:20.04
ENV PATH="${PATH}:/usr/lib/bashio"
RUN apt update 
    && apt install -y dos2unix python3 python3-pip \
    && pip install --upgrade pip \
    && pip install scikit-learn psutil func_timeout requests numpy pandas pyswarms tomli

### Addon Image Build ###
FROM narcisplan/scheduler_addon:v2
ENV PATH="${PATH}:/usr/lib/bashio"
# Copy root filesystem
COPY rootfs /
COPY config.yaml config.yaml
RUN pip install holidays 
# TODO: integrate "holidays" into base image
RUN dos2unix ./etc/services.d/example/Abstraction/*.py
# TODO: Try to convert all files into LF format because github converts them to CRLF and the addon does not work with CLF format
#RUN find /rootfs -type f -exec dos2unix {} \; 

