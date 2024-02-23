#!/usr/bin/env bashio

message=$(bashio::config 'message')
bashio::log.info "${message:="Hello World..."}"