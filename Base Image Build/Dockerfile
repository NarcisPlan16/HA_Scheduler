# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

#FROM ghcr.io/home-assistant/aarch64-base-python:latest
FROM ghcr.io/home-assistant/aarch64-base-ubuntu:20.04

ENV PATH="${PATH}:/usr/lib/bashio"

RUN apt update

# Install python
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN cd tmp && wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
RUN cd tmp && tar -xf Python-3.12.3.tgz
RUN cd tmp && Python-3.12.3/configure --enable-optimizations
#RUN make install

#RUN apt install g++ gfortran pkgconfig cmake openblas openblas-dev
#RUN pip install psutil func_timeout requests numpy pandas scikit-learn
#CAUSE PROBLEMS: scipy pyswarms

