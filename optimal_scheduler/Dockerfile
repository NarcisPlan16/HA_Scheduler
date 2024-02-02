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

#----------PYTHON----------#
FROM python:3.11.7-bookworm

# Install dependencies
RUN pip install matplotlib psutil pyswarms func_timeout customtkinter

# Copy Python script or application
COPY /Abstraction Abstraction

# Set the working directory
WORKDIR /Abstraction

# Command to run when the container starts
CMD ["python", "main.py"]