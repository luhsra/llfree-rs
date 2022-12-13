#!/bin/bash
set -e

################################################################################
# Benchmarks
################################################################################
export RUST_LOG=warn
OUT="results/out-$(date '+%y%m%d_%H%M')"
mkdir -p $OUT

ALLOC="Array3C32 Array3C64 Array3A128"
THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48"
FILL="-x0 -x10 -x20 -x30 -x40 -x50 -x60 -x70 -x80"
FLAGS="-t48 -i4 -m192 --stride 1"

echo "ALLOC=$ALLOC
FLAGS=$FLAGS
THREADS=$THREADS
KERNEL=$(uname -r)
HOST=$(hostname)" | tee $OUT/meta.txt

# DRAM
echo ">>> check v bulk"
for o in {0..10}
do
    target/release/bench bulk $ALLOC $FLAGS $THREADS \
        -o $OUT/bulk_v$o.csv -s$o
done
echo ">>> check v repeat"
for o in {0..10}
do
    target/release/bench repeat $ALLOC $FLAGS $THREADS \
        -o $OUT/repeat_v$o.csv -s$o
done
echo ">>> check v rand"
for o in {0..10}
do
    target/release/bench rand $ALLOC $FLAGS $THREADS \
        -o $OUT/rand_v$o.csv -s$o
done
# echo ">>> check p rand-block"
# target/release/bench rand-block $FLAGS $ALLOC $THREADS \
#     -o $OUT/rand_block_v.csv

exit 0;

# NVRAM
FLAGS="$FLAGS --dax=/dev/dax0.0"

echo ">>> check p bulk"
target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o $OUT/bulk_p.csv
echo ">>> check p repeat"
target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o $OUT/repeat_p.csv
echo ">>> check p rand"
target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o $OUT/rand_p.csv
# echo ">>> check p rand-block"
# target/release/bench rand-block $FLAGS $ALLOC $THREADS \
#     -o $OUT/rand_block_p.csv
