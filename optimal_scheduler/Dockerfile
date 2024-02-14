# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Execute during the build of the image
ARG TEMPIO_VERSION BUILD_ARCH
RUN \
    curl -sSLf -o /usr/bin/tempio \
    "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}"

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
CMD ["/bin/ls", "-l"] 
CMD ["run.sh"]