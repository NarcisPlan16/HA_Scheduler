# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Execute during the build of the image
# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash

ARG TEMPIO_VERSION BUILD_ARCH BASHIO_VERSION

# Install tempio
RUN curl -sSLf -o /usr/bin/tempio "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}"

# Create directory for bashio
RUN mkdir -p /usr/src/bashio

# Download bashio
RUN curl -L -f -s "https://github.com/hassio-addons/bashio/archive/v0.16.2.tar.gz" | \
    tar -xzf - --strip 1 -C /usr/src/bashio

# Move bashio library
RUN mv /usr/src/bashio/lib /usr/lib/bashio



# Create symbolic link for bashio
#RUN ln -s /usr/lib/bashio/bashio /usr/bin/bashio

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

RUN chmod +x run.sh

# Set the working directory
#WORKDIR /usr/local/bin

# Command to run when the container starts
#CMD ["python", "Abstraction/main.py"] 
#CMD ["/bin/ls", "-l"] 
RUN ["bash", "-c", "ls -l /bin/lib | grep bashio"]
#CMD ["sh", "./run.sh"]