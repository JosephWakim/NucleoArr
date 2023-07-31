#!/bin/bash

yes | conda create --name slide python=3.9.12
eval "$(conda shell.bash hook)"
conda activate slide
yes | pip install -r requirements.txt
pip install -e .