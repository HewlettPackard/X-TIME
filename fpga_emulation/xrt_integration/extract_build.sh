#!/bin/bash

cd $(dirname "$0")

md5sum build.tar.gz
tar -xvf build.tar.gz
