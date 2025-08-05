#!/bin/bash
docker rm -f blazegraph
docker run -d --name blazegraph -p 9999:8080 lyrasis/blazegraph:2.1.4
python scripts/blazegraph_setup.py  # run from project root
