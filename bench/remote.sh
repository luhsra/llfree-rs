export RUST_LOG=warn
mkdir -p bench/out
set -e

ALLOC="ArrayAtomicA128 ArrayAtomicA256 ArrayAtomicA512 ArrayListA128 ArrayListA256 ArrayListA512"
THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48 -x56 -x64 -x80 -x96"
FILL="-x0 -x10 -x20 -x30 -x40 -x50 -x60 -x70 -x80"

# NVRAM
FLAGS="-t96 -i4 -s0 -m192 --dax=/dev/dax0.0 --stride 2"

echo ">>> check p bulk"
chrt 99 target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o results/out/bulk_p.csv
echo ">>> check p repeat"
chrt 99 target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o results/out/repeat_p.csv
echo ">>> check p rand"
chrt 99 target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o results/out/rand_p.csv

# DRAM
FLAGS="-t96 -i4 -s0 -m192 --stride 2"

echo ">>> check v bulk"
chrt 99 target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o results/out/bulk_v.csv
echo ">>> check v repeat"
chrt 99 target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o results/out/repeat_v.csv
echo ">>> check v rand"
chrt 99 target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o results/out/rand_v.csv

exit 0;

# NVRAM
FLAGS="-t96 -i4 -s9 -m512 --dax=/dev/dax0.1"

echo ">>> check p bulk1"
chrt 99 target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o results/out/bulk_p1.csv
echo ">>> check p repeat1"
chrt 99 target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o results/out/repeat_p1.csv
echo ">>> check p rand1"
chrt 99 target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o results/out/rand_p1.csv

# DRAM
FLAGS="-t96 -i4 -s9 -m192"

echo ">>> check v bulk1"
chrt 99 target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o results/out/bulk_v1.csv
echo ">>> check v repeat1"
chrt 99 target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o results/out/repeat_v1.csv
echo ">>> check v rand1"
chrt 99 target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o results/out/rand_v1.csv
