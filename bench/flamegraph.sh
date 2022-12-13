FLAMEGRAPH_PATH=/srv/scratch/wrenger/FlameGraph

OUT="results/flamegraph_$(date '+%y%m%d_%H%M')"
mkdir -p $OUT

perf record -F 99 -g -- $@
perf script > $OUT/out.perf

$FLAMEGRAPH_PATH/stackcollapse-perf.pl --all $OUT/out.perf > $OUT/out.folded
$FLAMEGRAPH_PATH/flamegraph.pl $OUT/out.folded > $OUT/out.svg
