# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

FROM narcisplan/scheduler_addon:v2

ENV PATH="${PATH}:/usr/lib/bashio"

# Copy root filesystem
COPY rootfs /
COPY config.yaml config.yaml

RUN dos2unix ./etc/services.d/example/Abstraction/*.py

# TODO: Try to convert all files into LF format because github converts them to CRLF and the addon does not work with CLF format
#RUN find /rootfs -type f -exec dos2unix {} \; 

