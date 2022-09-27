LINUX_BUILD_DIR ?= $(PWD)/../../nvalloc-linux

obj-m += alloc.o

all:
	ln -s $(LINUX_BUILD_DIR)/.clang-format .clang-format
	$(LINUX_BUILD_DIR)/scripts/clang-tools/gen_compile_commands.py
	make -C $(LINUX_BUILD_DIR) M=$(PWD) modules

clean:
	make -C $(LINUX_BUILD_DIR) M=$(PWD) clean
	rm -f .clang-format compile_commands.json
