#!/bin/bash

run-omb-py(){
	if [ "$#" -eq 0 ]; then
		echo "No Arguments Provided - Please Enter: Number of Processes (integer), Benchmark (string), Buffer (string), Iterations (integer)"
		return 0
	fi
	dt=$(date '+%m%d%Y-%H%M%S')
	mkdir "$2-$dt" 
	mpirun -np $1 python osu-micro-benchmarks-7.4/python/run.py --benchmark $2 --buffer $3 --iterations $4 >> "$2-$dt/omb-$2-$HOSTNAME"
}
