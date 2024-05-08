# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile

FROM ghcr.io/home-assistant/aarch64-base-python:latest

ENV PATH="${PATH}:/usr/lib/bashio"

RUN apk update && apk upgrade
RUN apk add g++ gfortran pkgconfig cmake openblas openblas-dev
RUN pip install psutil func_timeout requests numpy pandas
RUN apk add py3-scikit-learn
#CAUSE PROBLEMS: scipy pyswarms

