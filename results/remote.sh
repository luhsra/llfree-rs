export RUST_LOG=warn
mkdir -p bench/out

#ALLOC="ArrayAlignedD ArrayAlignedF ArrayAlignedP ArrayAtomicD ArrayAtomicF ArrayAtomicP ArrayLockedD ArrayLockedF ArrayLockedP TableD TableF TableP ListLocal"
ALLOC="ArrayAtomicP ArrayAtomicC64 ArrayAtomicC128 ArrayAtomicC256 ArrayAtomicC512 ArrayAlignedP ArrayAlignedC64 ArrayAlignedC128 ArrayAlignedC256 ArrayAlignedC512"
THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48 -x56 -x64 -x80 -x96"
FILL="-x0 -x10 -x20 -x30 -x40 -x50 -x60 -x70 -x80"

# NVRAM
FLAGS="-t96 -i4 -s0 -m256 --dax=/dev/dax0.1 --stride 2"

#echo ">>> check p bulk"
chrt 99 target/release/examples/bench bulk $ALLOC $FLAGS $THREADS \
    -o bench/out/bulk_p.csv
echo ">>> check p repeat"
chrt 99 target/release/examples/bench repeat $ALLOC $FLAGS $THREADS \
    -o bench/out/repeat_p.csv
echo ">>> check p rand"
chrt 99 target/release/examples/bench rand $FLAGS $ALLOC $THREADS \
    -o bench/out/rand_p.csv
#echo ">>> check p filling"
#chrt 99 target/release/examples/bench filling -t24 -i4 -s0 -m240 --dax=/dev/dax0.1 $ALLOC $FILL \
#    -o bench/out/filling_p.csv

#echo ">>> check p align"
#chrt 99 target/release/examples/bench bulk ArrayAlignedD ArrayUnalignedD $FLAGS $THREADS \
#    -o bench/out/align_p.csv

# DRAM
FLAGS="-t96 -i4 -s0 -m256 --stride 2"

echo ">>> check v bulk"
chrt 99 target/release/examples/bench bulk $ALLOC $FLAGS $THREADS \
    -o bench/out/bulk_v.csv
echo ">>> check v repeat"
chrt 99 target/release/examples/bench repeat $ALLOC $FLAGS $THREADS \
    -o bench/out/repeat_v.csv
echo ">>> check v rand"
chrt 99 target/release/examples/bench rand $FLAGS $ALLOC $THREADS \
    -o bench/out/rand_v.csv
#echo ">>> check v filling"
#chrt 99 target/release/examples/bench filling -t24 -i4 -s0 -m240 $ALLOC $FILL \
#    -o bench/out/filling_v.csv

#echo ">>> check v align"
#chrt 99 target/release/examples/bench bulk ArrayAlignedD ArrayUnalignedD $FLAGS $THREADS \
#    -o bench/out/align_v.csv

exit 0;

# NVRAM
FLAGS="-t96 -i4 -s1 -m512 --dax=/dev/dax0.1"

echo ">>> check p bulk1"
chrt 99 target/release/examples/bench bulk $ALLOC $FLAGS $THREADS \
    -o bench/out/bulk_p1.csv
echo ">>> check p repeat1"
chrt 99 target/release/examples/bench repeat $ALLOC $FLAGS $THREADS \
    -o bench/out/repeat_p1.csv
echo ">>> check p rand1"
chrt 99 target/release/examples/bench rand $FLAGS $ALLOC $THREADS \
    -o bench/out/rand_p1.csv

# DRAM
FLAGS="-t96 -i4 -s1 -m192"

echo ">>> check v bulk1"
chrt 99 target/release/examples/bench bulk $ALLOC $FLAGS $THREADS \
    -o bench/out/bulk_v1.csv
echo ">>> check v repeat1"
chrt 99 target/release/examples/bench repeat $ALLOC $FLAGS $THREADS \
    -o bench/out/repeat_v1.csv
echo ">>> check v rand1"
chrt 99 target/release/examples/bench rand $FLAGS $ALLOC $THREADS \
    -o bench/out/rand_v1.csv



# NVRAM
FLAGS="-t64 -i4 -s2 -m512 --dax=/dev/dax0.1"
THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48 -x56 -x64"

echo ">>> check p bulk2"
chrt 99 target/release/examples/bench bulk $ALLOC $FLAGS $THREADS \
    -o bench/out/bulk_p2.csv
echo ">>> check p repeat2"
chrt 99 target/release/examples/bench repeat $ALLOC $FLAGS $THREADS \
    -o bench/out/repeat_p2.csv
echo ">>> check p rand2"
chrt 99 target/release/examples/bench rand $FLAGS $ALLOC $THREADS \
    -o bench/out/rand_p2.csv

# DRAM
FLAGS="-t64 -i4 -s2 -m256"

echo ">>> check v bulk2"
chrt 99 target/release/examples/bench bulk $ALLOC $FLAGS $THREADS \
    -o bench/out/bulk_v2.csv
echo ">>> check v repeat2"
chrt 99 target/release/examples/bench repeat $ALLOC $FLAGS $THREADS \
    -o bench/out/repeat_v2.csv
echo ">>> check v rand2"
chrt 99 target/release/examples/bench rand $FLAGS $ALLOC $THREADS \
    -o bench/out/rand_v2.csv
