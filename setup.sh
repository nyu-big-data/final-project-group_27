#!/bin/bash
export PYTHONPATH=/${SCRATCH}/final-project-group_27/code/
export PROJECT_ROOT=$(pwd)
export HADOOP_EXE='/usr/bin/hadoop'
module load python/gcc/3.7.9
module load spark/3.0.1

alias hfs="$HADOOP_EXE fs"
alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit'