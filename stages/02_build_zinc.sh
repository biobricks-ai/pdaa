#!/usr/bin/env bash

#define path
localpath=$(pwd)
echo "Local path: $localpath"

downloadpath="$localpath/download"
echo "Download path: $downloadpath"

# run conversion to parquet
mkdir -p $localpath/raw
python ./stages/zinc2parquet.py $downloadpath/ $localpath/raw/zinc.parquet