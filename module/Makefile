# In kbuild context
obj-y := nvalloc.o
nvalloc-objs := nvalloc_c.o nvalloc_rs.o

# In normal make context
NVALLOC := /workspaces/nvalloc-rs/target/x86_64-linux/release/libnvalloc_module.a
KERNELDIR ?= ../../linux
PWD := $(shell pwd)

.PHONY: $(NVALLOC)
$(NVALLOC):
	RUSTFLAGS="-Zno-unique-section-names -Ccodegen-units=1 -Crelocation-model=static -Zfunction-sections=n -Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-avx,-avx2 -Ztune-cpu=generic -Cno-redzone=y -Ccode-model=kernel -Ctarget-feature=+retpoline-external-thunk -Cdebug-assertions=n -Coverflow-checks=y -Copt-level=2" cargo build -r --target=x86_64-linux.json -Zbuild-std=core,alloc,compiler_builtins

nvalloc_rs.o: $(NVALLOC)
	ld.lld -r -m elf_x86_64 --lto-O3 --thinlto-jobs=4 --whole-archive $^ -o $@

.PHONY: modules
modules: nvalloc_rs.o
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

.PHONY: clean
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
	rm -f $(NVALLOC)
