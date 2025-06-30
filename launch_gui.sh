#!/bin/bash

#if you are using a conda environment to run the script (recommended) uncomment these and use your conda environment
#change this to the path to your conda environment
#source "/home/user/anaconda3/etc/profile.d/conda.sh"
#change this to your conda environment
#conda activate torch_env

#change this to the path to your Colony_analyzer_gui.py file
python /home/user/.../Colony_analyzer_gui.py

# Debugging: wait if script fails
if [ $? -ne 0 ]; then
  echo "Script failed. Press enter to close."
  read
fi
