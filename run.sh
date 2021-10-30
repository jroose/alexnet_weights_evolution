#!/bin/bash

source /data/collab/src/alexnet_weights_evolution/venv/bin/activate
python3 main.py /data/collab/data/ILSVRC/Data/CLS-LOC/{train,val,test}
