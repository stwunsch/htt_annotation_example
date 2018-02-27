#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc62-opt/setup.sh

export KERAS_BACKEND=theano
export OMP_NUM_THREADS=8
export THEANO_FLAGS=gcc.cxxflags=-march=corei7

FILE=$1
TAG=$2
python annotate_file.py $FILE $TAG
