# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
FROM ghcr.io/home-assistant/armhf-base-raspbian:latest

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Execute during the build of the image
WORKDIR /usr/src

RUN \
    set -x \
    && apt-get update && apt-get install -y --no-install-recommends \
        bash \
        jq \
        tzdata \
        curl \
        ca-certificates \
        xz-utils

RUN mkdir -p /usr/src/bashio
RUN curl -L -f -s "https://github.com/hassio-addons/bashio/archive/v0.16.2.tar.gz" | tar -xzf - --strip 1 -C /usr/src/bashio
RUN mv /usr/src/bashio/lib /usr/lib/bashio
RUN ln -s /usr/lib/bashio/bashio /usr/bin/bashio
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /usr/src/*

#RUN cd .. && cd ..
WORKDIR /

# Copy root filesystem
COPY rootfs /

#####################################################
# TODO: Fix github action build architecture failure
#RUN apt-get update \
#    && apt-get install -y --no-install-recommends \
#        gcc \
#        gfortran
#####################################################

#----------PYTHON----------#
FROM python:3.11.7-bookworm

# Install dependencies
RUN pip install psutil pyswarms func_timeout requests
# matplotlib

# Copy Python script or application
COPY /Abstraction Abstraction
COPY config.yaml config.yaml
COPY run.sh run.sh
COPY findBashio.sh findBashio.sh

RUN chmod 777 run.sh
RUN chmod 777 findBashio.sh

# Command to run when the container starts
#CMD ["python", "Abstraction/main.py"]
#CMD ["sh", "./run.sh"]
#CMD ["sh", "./findBashio.sh"]