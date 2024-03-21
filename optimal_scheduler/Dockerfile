# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

FROM ghcr.io/home-assistant/aarch64-base-python:latest

ENV PATH="${PATH}:/usr/lib/bashio"
#ENV PATH="${PATH}:/usr/python"

# Copy root filesystem
COPY rootfs /

RUN dos2unix ./etc/services.d/example/Abstraction/*.py

COPY config.yaml config.yaml

RUN pip install psutil func_timeout requests numpy
#CAUSE PROBLEMS: scipy pyswarms

# Command to run when the container starts
#CMD ["python", "Abstraction/main.py"]
#CMD ["sh", "./run.sh"]
