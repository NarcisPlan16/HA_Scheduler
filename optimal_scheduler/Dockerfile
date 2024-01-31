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

RUN ls -l

#----------PYTHON----------
FROM python:3.8-alpine

# Install dependencies
#RUN pip install your_dependency

# Copy your Python script or application
#COPY main.py /app/main.py

# Set the working directory
WORKDIR /app

# Command to run when the container starts
#CMD ["python", "main.py"]