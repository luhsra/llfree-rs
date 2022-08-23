#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/completion.h>
#include <linux/delay.h>
#include <linux/gfp.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/timekeeping.h>

#include "nanorand.h"

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel Alloc Test");
MODULE_AUTHOR("Lars Wrenger");

#ifndef NUM_ALLOCS
#define NUM_ALLOCS (512 * 512)
#endif

#ifndef THREADS_MAX
#define THREADS_MAX 6UL
#endif

#ifndef ITERATIONS
#define ITERATIONS 4
#endif

static const u64 threads[] = { 1,  2,  4,  8,  16, 20, 24,
			       32, 40, 48, 56, 64, 80, 96 };
#define THREADS_LEN (sizeof(threads) / sizeof(*threads))

static struct task_struct *tasks[THREADS_MAX];
static DECLARE_COMPLETION(start_barrier);
static DECLARE_COMPLETION(mid_barrier);
static struct completion barriers[THREADS_MAX];
static bool running = false;

struct thread_perf {
	atomic64_t get;
	atomic64_t put;
};
static struct thread_perf thread_perf[THREADS_MAX];

struct perf {
	u64 get_min;
	u64 get_avg;
	u64 get_max;
	u64 put_min;
	u64 put_avg;
	u64 put_max;
};
static struct perf perf[ITERATIONS * THREADS_LEN];
static u64 out_index = 0;

struct work_data {
	union {
		struct {
			u32 tid;
			u32 bench;
		};
		u64 value;
	};
} __packed;

__maybe_unused static u64 cycles(void)
{
	u32 lo, hi;
	asm volatile("rdtsc" : "=eax"(lo), "=edx"(hi) :);
	return ((u64)lo) | ((u64)hi) << 32;
};

/// Alloc a number of pages at once and free them afterwards
__maybe_unused static void bulk(u64 tid)
{
	u64 j;
	u64 timer;
	struct page **pages =
		kmalloc_array(NUM_ALLOCS, sizeof(struct page *), GFP_KERNEL);

	if (pages == NULL) {
		pr_err("kmalloc failed");
		return;
	}

	// complete initialization
	complete(&barriers[tid]);

	// Start allocations
	wait_for_completion(&start_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < NUM_ALLOCS; j++) {
		pages[j] = alloc_page(GFP_USER);
		if (pages == NULL) {
			pr_err("alloc_page failed");
		}
	}
	timer = (ktime_get_ns() - timer) / NUM_ALLOCS;
	atomic64_set(&thread_perf[tid].get, timer);
	pr_info("Alloc %llu\n", timer);

	complete(&barriers[tid]);

	// Start frees
	wait_for_completion(&mid_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < NUM_ALLOCS; j++) {
		__free_page(pages[j]);
	}
	timer = (ktime_get_ns() - timer) / NUM_ALLOCS;
	atomic64_set(&thread_perf[tid].put, timer);
	pr_info("Free %llu\n", timer);
	kfree(pages);
}

/// Alloc and free the same page
__maybe_unused static void repeat(u64 tid)
{
	u64 j;
	u64 timer;

	struct page *page;

	// complete initialization
	complete(&barriers[tid]);

	// Start reallocs
	wait_for_completion(&mid_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < NUM_ALLOCS; j++) {
		page = alloc_page(GFP_USER);
		if (page == NULL) {
			pr_err("alloc_page failed");
		}
		__free_page(page);
	}
	timer = (ktime_get_ns() - timer) / NUM_ALLOCS;
	atomic64_set(&thread_perf[tid].get, timer);
	atomic64_set(&thread_perf[tid].put, timer);
	pr_info("Realloc %llu\n", timer);
}

/// Random free and realloc
__maybe_unused static void rand(u64 tid)
{
	u64 i, j;
	u64 timer;
	u64 rng = tid;

	struct page **pages =
		kmalloc_array(NUM_ALLOCS, sizeof(struct page *), GFP_KERNEL);
	if (pages == NULL) {
		pr_err("kmalloc failed");
		return;
	}

	for (j = 0; j < NUM_ALLOCS; j++) {
		pages[j] = alloc_page(GFP_USER);
		if (pages == NULL) {
			pr_err("alloc_page failed");
		}
	}

	// complete initialization
	complete(&barriers[tid]);

	// Start reallocs
	wait_for_completion(&mid_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < NUM_ALLOCS; j++) {
		i = nanorand_random_range(&rng, 0, NUM_ALLOCS);

		__free_page(pages[i]);
		pages[i] = alloc_page(GFP_USER);
		if (pages[i] == NULL) {
			pr_err("alloc_page failed");
		}
	}
	timer = (ktime_get_ns() - timer) / NUM_ALLOCS;
	atomic64_set(&thread_perf[tid].get, timer);
	atomic64_set(&thread_perf[tid].put, timer);

	for (j = 0; j < NUM_ALLOCS; j++) {
		__free_page(pages[j]);
	}

	pr_info("Realloc %llu\n", timer);
	kfree(pages);
}

static int worker(void *data)
{
	struct work_data d = { .value = (u64)data };

	pr_info("Worker %u bench %u\n", d.tid, d.bench);

	if (d.bench == 0) {
		bulk(d.tid);
	} else if (d.bench == 1) {
		repeat(d.tid);
	} else if (d.bench == 2) {
		rand(d.tid);
	}

	complete(&barriers[d.tid]);

	return 0;
}

/// Outputs the measured data.
/// Note: `buf` is PAGE_SIZE large!
static ssize_t out_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	struct perf *p;
	ssize_t len = 0;

	if (running)
		return -EINPROGRESS;

	if (out_index == 0) {
		len += sprintf(buf,
			       "alloc,x,iteration,allocs,get_min,get_avg,"
			       "get_max,put_min,put_avg,put_max,init,total\n");
	}

	for (ssize_t i = out_index;
	     i < THREADS_LEN && threads[i] <= THREADS_MAX; i++) {
		// The output buffer has only the size of a PAGE.
		// If our output is larger we have to output it in multiple steps.
		if (len < PAGE_SIZE - ITERATIONS * 128) {
			for (ssize_t iter = 0; iter < ITERATIONS; iter++) {
				p = &perf[i * ITERATIONS + iter];

				len += sprintf(
					buf + len,
					"Kernel,%llu,%lu,%llu,%llu,%llu,%llu,%llu,"
					"%llu,%llu,0,0\n",
					threads[i], iter, (u64)NUM_ALLOCS,
					p->get_min, p->get_avg, p->get_max,
					p->put_min, p->put_avg, p->put_max);
			}
		} else {
			out_index = i;
			return len;
		}
	}
	out_index = 0;
	return len;
}

ssize_t run_store(struct kobject *kobj, struct kobj_attribute *attr,
		  const char *buf, size_t len)
{
	u32 bench = 0;

	if (running)
		return -EINPROGRESS;
	running = true;

	if (len == 0 || len > 1 || buf == NULL)
		return -EINVAL;

	if (strncmp(buf, "bulk", 1)) {
		bench = 0;
	} else if (strncmp(buf, "repeat", 1)) {
		bench = 1;
	} else if (strncmp(buf, "rand", 1)) {
		bench = 2;
	} else {
		return -EINVAL;
	}

	for (u64 i = 0; i < THREADS_LEN && threads[i] <= THREADS_MAX; i++) {
		for (u64 iter = 0; iter < ITERATIONS; iter++) {
			struct perf *p;
			u64 get, put;

			pr_info("Start threads %llu\n", threads[i]);
			for (u64 t = 0; t < threads[i]; t++) {
				struct work_data data = {
					.tid = t,
					.bench = bench,
				};
				tasks[t] = kthread_create(
					worker, (void *)data.value, "worker");
				if (IS_ERR(tasks[t])) {
					pr_err("Unable to init %llu\n", t);
					return PTR_ERR(tasks[t]);
				}
				kthread_bind(tasks[t], t);
				init_completion(&barriers[t]);
				wake_up_process(tasks[t]);
			}

			// Init
			for (u64 t = 0; t < threads[i]; t++) {
				wait_for_completion(&barriers[t]);
				reinit_completion(&barriers[t]);
			}

			pr_info("Exec %llu threads\n", threads[i]);
			p = &perf[i * ITERATIONS + iter];
			p->get_min = (u64)-1;
			p->get_avg = 0;
			p->get_max = 0;
			p->put_min = (u64)-1;
			p->put_avg = 0;
			p->put_max = 0;

			// bulk alloc has two phases (alloc and free)
			if (bench == 0) {
				complete_all(&start_barrier);

				pr_info("Waiting for workers...\n");

				for (u64 t = 0; t < threads[i]; t++) {
					wait_for_completion(&barriers[t]);
					reinit_completion(&barriers[t]);
				}
			}

			// second phase
			complete_all(&mid_barrier);

			for (u64 t = 0; t < threads[i]; t++) {
				wait_for_completion(&barriers[t]);
				reinit_completion(&barriers[t]);
				get = atomic64_read(&thread_perf[t].get);
				put = atomic64_read(&thread_perf[t].put);

				p->get_min = min(p->get_min, get);
				p->get_avg += get;
				p->get_max = max(p->get_max, get);
				p->put_min = min(p->put_min, put);
				p->put_avg += put;
				p->put_max = max(p->put_max, put);
			}
			p->get_avg /= threads[i];
			p->put_avg /= threads[i];

			// Cleanup
			reinit_completion(&start_barrier);
			reinit_completion(&mid_barrier);
			for (u64 t = 0; t < threads[i]; t++) {
				kthread_stop(tasks[t]);
			}
		}
	}

	pr_info("Finished\n");
	running = false;

	return 0;
}

static struct kobj_attribute out_attribute = __ATTR(out, 0444, out_show, NULL);
static struct kobj_attribute run_attribute = __ATTR(out, 0220, NULL, run_store);

static struct attribute *attrs[] = {
	&out_attribute.attr, &run_attribute.attr,
	NULL, /* need to NULL terminate the list of attributes */
};

static struct attribute_group attr_group = {
	.attrs = attrs,
};
static struct kobject *output;

static int alloc_init_module(void)
{
	int retval;
	pr_info("Init\n");

	output = kobject_create_and_add(KBUILD_MODNAME, kernel_kobj);
	if (!output) {
		pr_err("KObj failed\n");
		return -ENOMEM;
	}

	retval = sysfs_create_group(output, &attr_group);
	if (retval) {
		pr_err("Sysfs failed\n");
		kobject_put(output);
	}

	return 0;
}

static void alloc_cleanup_module(void)
{
	pr_info("End\n");
	kobject_put(output);
}

module_init(alloc_init_module);
module_exit(alloc_cleanup_module);
