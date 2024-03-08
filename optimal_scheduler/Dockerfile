# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

FROM ghcr.io/home-assistant/aarch64-base-python:latest

ENV PATH="${PATH}:/usr/lib/bashio"
#ENV PATH="${PATH}:/usr/python"

# Copy root filesystem
COPY rootfs /
COPY config.yaml config.yaml

RUN chmod +x rootfs/usr/bin/Abstraction/main.py

RUN pip install psutil func_timeout requests 
#pyswarms

# Copy Python script or application
#COPY /Abstraction Abstraction

# Command to run when the container starts
#CMD ["python", "Abstraction/main.py"]
#CMD ["sh", "./run.sh"]
#CMD ["sh", "./findBashio.sh"]