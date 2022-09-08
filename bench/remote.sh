set -e

################################################################################
# Tune Performance
################################################################################

MIN_CPU=$(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct)

cleanup()
{
    echo reenable powersaving
    echo powersave | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    echo $MIN_CPU >> /sys/devices/system/cpu/intel_pstate/min_perf_pct
}
trap cleanup EXIT HUP INT QUIT KILL SEGV TERM

echo disable powersaving
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
echo 100 >> /sys/devices/system/cpu/intel_pstate/min_perf_pct

################################################################################
# Benchmarks
################################################################################
export RUST_LOG=warn
OUT=results/out
mkdir -p $OUT

ALLOC="ArrayAtomicA128 ArrayAtomicA256 ArrayAtomicA512 ArrayListA128 ArrayListA256 ArrayListA512"
# THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48 -x56 -x64 -x80 -x96"
THREADS="-x1 -x2 -x4 -x8 -x16 -x24 -x32 -x40 -x48"
FILL="-x0 -x10 -x20 -x30 -x40 -x50 -x60 -x70 -x80"

# NVRAM
FLAGS="-t96 -i4 -s0 -m192 --dax=/dev/dax0.0 --stride 1"

echo ">>> check p bulk"
chrt 99 target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o $OUT/bulk_p.csv
echo ">>> check p repeat"
chrt 99 target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o $OUT/repeat_p.csv
echo ">>> check p rand"
chrt 99 target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o $OUT/rand_p.csv

# DRAM
FLAGS="-t96 -i4 -s0 -m192 --stride 1"

echo ">>> check v bulk"
chrt 99 target/release/bench bulk $ALLOC $FLAGS $THREADS \
    -o $OUT/bulk_v.csv
echo ">>> check v repeat"
chrt 99 target/release/bench repeat $ALLOC $FLAGS $THREADS \
    -o $OUT/repeat_v.csv
echo ">>> check v rand"
chrt 99 target/release/bench rand $FLAGS $ALLOC $THREADS \
    -o $OUT/rand_v.csv
