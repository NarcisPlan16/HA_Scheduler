#!/usr/bin/with-contenv bashio

message=$(bashio::config 'message')
bashio::log.info "${message:="Hello World..."}"