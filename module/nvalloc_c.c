#include "nvalloc.h"

#include <linux/gfp.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

MODULE_LICENSE("MIT");
MODULE_DESCRIPTION("NVM Allocator");
MODULE_AUTHOR("Lars Wrenger");

#define MOD KBUILD_MODNAME ": "

/// Returns 0 on success or an error code
u64 nvalloc_init(u32 cores, void *addr, u64 pages, u32 overwrite);
void nvalloc_uninit(void);

// Functions needed by the allocator

/// Linux provided alloc function
u8 *nvalloc_linux_alloc(u64 size, u64 align)
{
	return kmalloc(size, GFP_KERNEL);
}
/// Linux provided free function
void nvalloc_linux_free(u8 *ptr, u64 size, u64 align)
{
	kfree(ptr);
}
/// Linux provided printk function
void nvalloc_printk(const u8 *format, const u8 *module_name, const void *args)
{
	_printk(format, module_name, args);
}

static int __init nvalloc_init_module(void)
{
	int cores, cpu;
	s64 ret;
	void *addr;
	u64 pages;
	void *mem;
	pr_info(MOD "init\n");
	cores = num_online_cpus();
	pr_info(MOD "cores %d\n", cores);

	pages = 512 * 512;
	mem = vmalloc_huge(pages * PAGE_SIZE, GFP_KERNEL);
	if (IS_ERR(mem))
	{
		pr_err(MOD "Failed memory allocation\n");
		return PTR_ERR(mem);
	}

	pr_info(MOD "mem 0x%llx l=%lld\n", (u64)mem, pages);

	ret = nvalloc_init(cores, mem, pages, true);
	pr_info(MOD "init ret=%d\n", ret);
	if (nvalloc_err(ret))
		return -ENOMEM;

	pr_info(MOD "try allocation");

	cpu = get_cpu();
	addr = nvalloc_get(cpu, 0);
	if (nvalloc_err((u64)(addr)))
	{
		pr_info(MOD "error alloc %ld\n", (u64)addr);
		put_cpu();
		return -ENOMEM;
	}
	pr_info(MOD "allocated %p on %d\n", addr, cpu);

	ret = nvalloc_put(cpu, addr, 0);
	if (nvalloc_err(ret))
	{
		pr_info(MOD "error free %ld\n", ret);
		put_cpu();
		return -ENOMEM;
	}

	put_cpu();
	pr_info(MOD "success\n");
	return 0;
}

static void nvalloc_cleanup_module(void)
{
	pr_info(MOD "uninit\n");
}

module_init(nvalloc_init_module);
module_exit(nvalloc_cleanup_module);
