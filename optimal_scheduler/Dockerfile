# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

#FROM ghcr.io/home-assistant/armhf-base-raspbian:latest
#FROM ghcr.io/home-assistant/armhf-base-python:latest
#FROM ghcr.io/home-assistant/armhf-base-debian:latest
FROM ghcr.io/home-assistant/aarch64-base-debian:latest

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
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
RUN bash Miniconda3-latest-Linux-aarch64.sh -b -p /opt/conda
RUN rm Miniconda3-latest-Linux-aarch64.sh  # Clean up the installer
#RUN exec $SHELL

ENV PATH="/opt/conda/bin:${PATH}"

# Create a new conda environment
RUN conda create -n myenv python

# Activate the new environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install Python libraries
RUN conda install -y psutil func_timeout requests
RUN conda install -y conda-forge::pyswarms

# Switch back to the default shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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