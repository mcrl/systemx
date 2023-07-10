#!/bin/bash

# TARGET=aluCompute
SYSX=~/systemx/build/systemx
DIR=~/systemx/benchmarks

for TARGET in aluCompute l2Load l2Store gmemLoad gmemStore
do
  for BENCHMARK in aluCompute l2Load l2Store gmemLoad gmemStore
  do
    ~/systemx/run-nsys.sh $SYSX --benchmark_file $DIR/serial/serial-$TARGET,$BENCHMARK.json; 

    ~/systemx/run-nsys.sh $SYSX --benchmark_file $DIR/concurrent-colocated/colocated-$TARGET,$BENCHMARK.json; 

    ~/systemx/run-nsys.sh $SYSX --benchmark_file $DIR/concurrent-isolated/isolated-$TARGET,$BENCHMARK.json; 
  done
done