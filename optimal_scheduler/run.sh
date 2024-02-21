#!/usr/bin/with-contenv bashio

find / -name "bashio" -type f 2>/dev/null

if command -v bashio &> /dev/null; then
    echo "bashio is in the PATH"
else
    echo "bashio is not in the PATH"
    exit 1
fi

message=$(bashio::config 'message')
bashio::log.info "${message:="Hello World..."}"