#!/usr/bin/env bash
#source /home/ccoupe/miniconda3/etc/profile.d/conda.sh
#eval "$(conda shell.bash hook)"
#conda activate py3
source `which virtualenvwrapper.sh`
workon py3
cd /usr/local/lib/mlshapes/
python shape_server.py --port 4439
