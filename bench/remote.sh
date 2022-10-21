set -e

################################################################################
# Benchmarks
################################################################################
export RUST_LOG=warn
OUT="results/out-$(date '+%y%m%d_%H%M')"
mkdir -p $OUT

ALLOC="Array0A128 Array1A128 Array3A128 ArrayList0A128 ArrayList1A128 ArrayList3A128"
THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48"
FILL="-x0 -x10 -x20 -x30 -x40 -x50 -x60 -x70 -x80"
FLAGS="-t96 -i4 -s0 -m192 --stride 1"

echo "ALLOC=$ALLOC\nFLAGS=$FLAGS\nTHREADS=$THREADS\nKERNEL=$(uname -r)\nHOST=$(hostname)" | tee $OUT/meta.txt

# DRAM
echo ">>> check v bulk"
target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o $OUT/bulk_v.csv
echo ">>> check v repeat"
target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o $OUT/repeat_v.csv
echo ">>> check v rand"
target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o $OUT/rand_v.csv
# echo ">>> check p rand-block"
# target/release/bench rand-block $FLAGS $ALLOC $THREADS \
#     -o $OUT/rand_block_v.csv

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
