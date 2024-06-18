# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

#FROM ghcr.io/home-assistant/aarch64-base-python:latest
FROM ghcr.io/home-assistant/aarch64-base-ubuntu:20.04

ENV PATH="${PATH}:/usr/lib/bashio"

RUN apt update
RUN apt-get update
RUN apt-get install dos2unix
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y 

#RUN apt install g++ gfortran pkgconfig cmake openblas openblas-dev
RUN pip install --upgrade pip
RUN pip install scikit-learn psutil func_timeout requests numpy pandas pyswarms tomli

