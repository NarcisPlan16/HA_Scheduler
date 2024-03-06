# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

FROM ghcr.io/home-assistant/aarch64-base-python:latest

#ENV PATH="${PATH}:/usr/bin/bashio"
ENV PATH="${PATH}:/usr/lib/bashio"

# Copy root filesystem
COPY rootfs /

RUN pip install psutil func_timeout requests 
#pyswarms

# Copy Python script or application
COPY /Abstraction Abstraction
COPY config.yaml config.yaml
COPY run.sh run.sh
COPY findBashio.sh findBashio.sh

RUN chmod 777 run.sh
RUN chmod 777 findBashio.sh

RUN echo $PATH

# Command to run when the container starts
#CMD ["python", "Abstraction/main.py"]
#CMD ["sh", "./run.sh"]
#CMD ["sh", "./findBashio.sh"]