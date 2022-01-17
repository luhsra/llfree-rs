#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/gfp.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel Alloc Test");
MODULE_AUTHOR("Lars Wrenger");

#define MOD "[" KBUILD_MODNAME "]: "

#define NUM_ALLOCS (2 * 512 * 512)
#define CPU_STRIDE 2

#define threads_max 6
u64 threads[] = {1, 2, 4, threads_max};
#define threads_len (sizeof(threads) / sizeof(*threads))

struct task_struct *tasks[threads_max];
struct completion start_barrier;
struct completion barriers[threads_max];

static u64 cycles(void) {
    u32 lo, hi;
    asm volatile("rdtsc" : "=eax" (lo), "=edx" (hi) : );
    return ((u64)lo) | ((u64)hi) << 32;
};

static int worker(void *data) {
    u64 i;
    u64 t = (u64)data;
    u64 timer;
    struct page **pages = kmalloc_array(NUM_ALLOCS, sizeof(struct page *), GFP_KERNEL);

    printk(KERN_INFO MOD "Worker %llu\n", t);


    if (pages == NULL) {
        printk(KERN_ERR MOD "kmalloc failed");
        complete_all(&barriers[t]);
        return -1;
    }

    wait_for_completion(&start_barrier);

    timer = cycles();

    for (i = 0; i < NUM_ALLOCS; i++) {
        pages[i] = alloc_page(GFP_USER);
        if (pages == NULL) {
            printk(KERN_ERR MOD "alloc_page failed");
            complete_all(&barriers[t]);
            return -1;
        }
    }

    printk(KERN_INFO MOD "alloc %llu\n", (cycles() - timer) / NUM_ALLOCS);
    timer = cycles();

    for (i = 0; i < NUM_ALLOCS; i++) {
        __free_page(pages[i]);
    }

    printk(KERN_INFO MOD "free %llu\n", (cycles() - timer) / NUM_ALLOCS);

    printk(KERN_INFO MOD "Worker %llu finished\n", t);

    complete_all(&barriers[t]);
    return t;
}

static int alloc_test_init_module(void) {
    u64 i, t;

    printk(KERN_INFO MOD "Init\n");

    for (i = 0; i < threads_len; i++) {
        printk(KERN_INFO MOD "start threads %lld\n", threads[i]);

        init_completion(&start_barrier);

        for (t = 0; t < threads[i]; t++) {
            tasks[t] = kthread_create(worker, (void *)t, "worker");
            if (IS_ERR(tasks[t])) {
                printk(KERN_ERR MOD "Unable to init %llu\n", t);
                return PTR_ERR(tasks[t]);
            }
            kthread_bind(tasks[t], CPU_STRIDE * t);
            init_completion(&barriers[t]);
            wake_up_process(tasks[t]);
        }

        msleep(300);
        complete_all(&start_barrier);

        for (t = 0; t < threads[i]; t++) {
            wait_for_completion(&barriers[t]);
            printk(KERN_INFO MOD "stop %llu\n", t);
        }

        msleep(100);
    }

    printk(KERN_INFO MOD "Finished\n");

    return 0;
}

static void alloc_test_cleanup_module(void) { printk(KERN_INFO MOD "End\n"); }

module_init(alloc_test_init_module);
module_exit(alloc_test_cleanup_module);
