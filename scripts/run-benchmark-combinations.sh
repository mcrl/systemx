#!/bin/bash

# Script to experiment all combinations of the benchmarks

SYSX=~/systemx/build/systemx
DIR=~/systemx/benchmarks

for TARGET in aluCompute l1Load l1Store l2Load l2Store gmemLoad gmemStore pcieLoad pcieStore
do
  for BENCHMARK in aluCompute l1Load l1Store l2Load l2Store gmemLoad gmemStore pcieLoad pcieStore
  do
    ~/systemx/run-nsys.sh $SYSX --benchmark_file $DIR/serial/serial-$TARGET,$BENCHMARK.json; 

    ~/systemx/run-nsys.sh $SYSX --benchmark_file $DIR/concurrent-colocated/colocated-$TARGET,$BENCHMARK.json; 

    ~/systemx/run-nsys.sh $SYSX --benchmark_file $DIR/concurrent-isolated/isolated-$TARGET,$BENCHMARK.json; 
  done
done