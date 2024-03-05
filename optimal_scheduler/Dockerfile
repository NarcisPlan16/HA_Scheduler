# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

#FROM ghcr.io/home-assistant/armhf-base-raspbian:latest
#FROM ghcr.io/home-assistant/armhf-base-python:latest
#FROM ghcr.io/home-assistant/armhf-base-debian:latest
FROM ghcr.io/home-assistant/armv7-base-debian:latest

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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
# python:3.11.7-bookworm
RUN apt update && apt upgrade -y
RUN apt install -y curl bzip2
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-armv7l.sh
RUN bash Miniconda3-latest-Linux-armv7l.sh -b
RUN exec $SHELL

# Install dependencies
RUN conda install --yes psutil pyswarms func_timeout requests
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
CMD ["sh", "./run.sh"]
#CMD ["sh", "./findBashio.sh"]