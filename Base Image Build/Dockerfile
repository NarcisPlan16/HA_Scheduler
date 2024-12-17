# Usa una base que ja tingui Python preinstal·lat
FROM ghcr.io/home-assistant/aarch64-base-python:latest

# Configura el PATH per accedir a Bashio
ENV PATH="${PATH}:/usr/lib/bashio"

# Actualitza i instal·la dependències en una sola línia
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        dos2unix \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Actualitza pip
RUN pip install --upgrade pip

# Instal·la els paquets Python necessaris
RUN pip install scikit-learn psutil func_timeout requests numpy pandas pyswarms tomli

