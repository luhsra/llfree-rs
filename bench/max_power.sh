set -e

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

sudo nice -n -20 sudo -u wrenger $@
