#!/bin/bash

source setup.sh
#sudo killall verus_server sprout*
sudo killall rl-server client python

port=${1}
i=${2}                 #trace_file_index
target=${3}
tuning_period=${4}
latency=${5}
training=${6}
scheme=$7
qsize=100            # packets
initial_alpha=150    # it is scaled 100x
prefix=""

if [ $training -eq 2 ]
then
    prefix="results"
else
    prefix="training"
fi

./run-dcubic.sh $latency $port $i test $qsize $target $initial_alpha $tuning_period $training $scheme $prefix

