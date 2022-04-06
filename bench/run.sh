mkdir -p bench/out
export RUST_LOG=warn

ALLOCS="ArrayAtomicF ArrayAtomicP ArrayLockedF ArrayLockedP ArrayAlignedF ArrayAlignedP ArrayUnalignedF ArrayUnalignedP TableF TableP ListLocal ListLocked"
FLAGS="-t10 -m24 -i4"
THREADS="-x1 -x2 -x4 -x6 -x8 -x10"
FILL="-x0 -x10 -x20 -x30 -x40 -x50 -x60 -x70"

echo ">>> bench bulk"
target/release/examples/bench bulk $ALLOCS $THREADS $FLAGS -o bench/out/bulk.csv
echo ">>> bench repeat"
target/release/examples/bench repeat $ALLOCS $THREADS $FLAGS -o bench/out/repeat.csv
echo ">>> bench rand"
target/release/examples/bench rand $ALLOCS $THREADS $FLAGS -o bench/out/rand.csv
echo ">>> bench filling"
target/release/examples/bench filling $ALLOCS $FILL $FLAGS -o bench/out/filling.csv
echo ">>> bench align"
target/release/examples/bench bulk ArrayAlignedF ArrayUnalignedF ArrayAlignedD ArrayUnalignedD $THREADS $FLAGS -o bench/out/align.csv


echo ">>> bench bulk s1"
target/release/examples/bench bulk $ALLOCS $THREADS $FLAGS -s1 -o bench/out/bulk_s1.csv
echo ">>> bench repeat s1"
target/release/examples/bench repeat $ALLOCS $THREADS $FLAGS -s1 -o bench/out/repeat_s1.csv
echo ">>> bench rand s1"
target/release/examples/bench rand $ALLOCS $THREADS $FLAGS -s1 -o bench/out/rand_s1.csv

# FLAGS="-t6 -m24 -i4"
# THREADS="-x1 -x2 -x4 -x6"
# echo ">>> bench bulk s2"
# target/release/examples/bench bulk $ALLOCS $THREADS $FLAGS -s2 -o bench/out/bulk_s2.csv
# echo ">>> bench repeat s2"
# target/release/examples/bench repeat $ALLOCS $THREADS $FLAGS -s2 -o bench/out/repeat_s2.csv
# echo ">>> bench rand s2"
# target/release/examples/bench rand $ALLOCS $THREADS $FLAGS -s2 -o bench/out/rand_s2.csv
