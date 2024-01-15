#!/bin/bash
source /Users/ccoupe/.bash_profile
conda activate py38
cd /usr/local/lib/mlshapes/
python3 shape_server.py --port 4439
