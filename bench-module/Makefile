LINUX_BUILD_DIR ?= /lib/modules/$(shell uname -r)/build

obj-m += alloc.o

NUM_ALLOCS ?= '(2 * 512 * 512)'
CPU_STRIDE ?= 1
THREADS_MAX ?= 6
ITERATIONS ?= 4
BENCH ?= 0 # 0: bulk, 1: repeat, 2: rand
EXTRA_CFLAGS = -DNUM_ALLOCS=$(NUM_ALLOCS) -DCPU_STRIDE=$(CPU_STRIDE) -DTHREADS_MAX=$(THREADS_MAX) -DITERATIONS=$(ITERATIONS) -DBENCH=$(BENCH)

all:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) modules

clean:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) clean
