#!/bin/zsh
set -eu

export OBSDATE="16-01-21"

cd ${PYWIFES_HOME}/reduction_scripts
python generate_metadata_script.py configBlue.py ${WIFES_DATA_INPUT}/${OBSDATE}
python reduce_blue_data_python3.py configBlue.py  ${WIFES_DATA_INPUT}/${OBSDATE}
