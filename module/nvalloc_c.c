#include <linux/gfp.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("NVM Allocator");
MODULE_AUTHOR("Lars Wrenger");

#define MOD KBUILD_MODNAME ": "

// Functions provided

s64 nvalloc_init(u32 cores, void *addr, u64 pages, u32 overwrite);
void nvalloc_uninit(void);
s64 nvalloc_get(u32 core, u32 size);
s64 nvalloc_put(u32 core, u32 addr);

// Functions needed by the allocator!

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
	int cores, ret;
	u64 pages;
	void *mem;
	pr_info(MOD "init\n");
	cores = num_online_cpus();
	pr_info(MOD "cores %d\n", cores);

	pages = 512 * 512;
	mem = vmalloc_huge(pages * PAGE_SIZE, GFP_KERNEL);
	if (IS_ERR(mem)) {
		pr_err(MOD "Failed memory allocation\n");
		return PTR_ERR(mem);
	}

	pr_info(MOD "mem 0x%llx l=%lld", (u64)mem, pages);

	ret = nvalloc_init(cores, mem, pages, true);
	pr_info(MOD "init ret=%d\n", ret);
	return ret == 0 ? 0 : -ENOMEM;
}

static void nvalloc_cleanup_module(void)
{
	pr_info(MOD "uninit\n");
}

module_init(nvalloc_init_module);
module_exit(nvalloc_cleanup_module);
