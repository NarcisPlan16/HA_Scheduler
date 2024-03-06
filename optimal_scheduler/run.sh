#!/usr/bin/with-contenv bashio

CONFIG_PATH=/data/options.json

message="$(bashio::config 'message')"
#bashio::log.info "${message:="Hello World..."}"