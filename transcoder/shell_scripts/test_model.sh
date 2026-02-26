#!/bin/bash
# Quick test of the transcoder model

cd /usr/scratch/rmanimaran8/boltz
source boltz_env/bin/activate
cd transcoder
python model.py
