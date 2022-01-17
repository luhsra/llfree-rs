LINUX_BUILD_DIR ?= /lib/modules/$(shell uname -r)/build

obj-m += alloc_test.o

all:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) modules

clean:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) clean
