LINUX_BUILD_DIR ?= $(PWD)/../../nvalloc-linux

obj-m += alloc.o

all:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) modules

clean:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) clean
