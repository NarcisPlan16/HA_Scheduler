# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

#FROM ghcr.io/home-assistant/armhf-base-raspbian:latest
FROM ghcr.io/hassio-addons/addon-base-python:latest

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
#FROM python:3.11.7-bookworm

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
CMD ["sh", "./run.sh"]
#CMD ["sh", "./findBashio.sh"]