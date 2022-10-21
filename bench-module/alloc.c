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

static atomic64_t curr_threads;

static DEFINE_PER_CPU(struct task_struct *, per_cpu_tasks);

static DEFINE_PER_CPU(struct completion, worker_completes);
static DEFINE_PER_CPU(struct completion, worker_activates);

static DECLARE_COMPLETION(worker_barrier);
static DECLARE_COMPLETION(worker_barrier0);

enum alloc_bench {
	/// Allocate a large number of pages and free them in sequential order
	BENCH_BULK,
	/// Reallocate a single page and free it immediately over and over
	BENCH_REPEAT,
	/// Allocate a large number of pages and free them in random order
	BENCH_RAND,
};

/// Benchmark args
struct alloc_config {
	/// Benchmark (see enum alloc_bench)
	u64 bench;
	/// Array of thread counts
	u64 *threads;
	/// Len of threads array
	u64 threads_len;
	/// Number of repetitions
	u64 iterations;
	/// Number of allocations per thread
	u64 allocs;
	/// Size of the allocations
	u64 order;
};

static struct alloc_config alloc_config = { 0, NULL, 0, 0, 0, 0 };

static bool running = false;

struct thread_perf {
	u64 get;
	u64 put;
};
static DEFINE_PER_CPU(struct thread_perf, thread_perf);

struct perf {
	u64 get_min;
	u64 get_avg;
	u64 get_max;
	u64 put_min;
	u64 put_avg;
	u64 put_max;
};
static struct perf *measurements = NULL;
static u64 out_index = 0;

__maybe_unused static struct page ***rand_pages;

__maybe_unused static u64 cycles(void)
{
	u32 lo, hi;
	asm volatile("rdtsc" : "=eax"(lo), "=edx"(hi) :);
	return ((u64)lo) | ((u64)hi << 32);
};

/// Alloc a number of pages at once and free them afterwards
__maybe_unused static void bulk(u64 num_allocs)
{
	u64 j;
	u64 timer;
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct thread_perf *t_perf = this_cpu_ptr(&thread_perf);

	struct page **pages =
		kmalloc_array(num_allocs, sizeof(struct page *), GFP_KERNEL);
	BUG_ON(pages == NULL);

	// complete initialization
	complete(worker_complete);

	// Start allocations
	wait_for_completion(&worker_barrier0);

	timer = ktime_get_ns();
	for (j = 0; j < num_allocs; j++) {
		pages[j] = alloc_pages(GFP_USER, alloc_config.order);
		BUG_ON(pages[j] == NULL);
	}
	t_perf->get = (ktime_get_ns() - timer) / num_allocs;

	complete(worker_complete);

	// Start frees
	wait_for_completion(&worker_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < num_allocs; j++) {
		__free_pages(pages[j], alloc_config.order);
	}
	t_perf->put = (ktime_get_ns() - timer) / num_allocs;
	kfree(pages);
}

/// Alloc and free the same page
__maybe_unused static void repeat(u64 num_allocs)
{
	u64 j;
	u64 timer;
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct thread_perf *t_perf = this_cpu_ptr(&thread_perf);

	struct page *page;

	// complete initialization
	complete(worker_complete);

	// Start reallocs
	wait_for_completion(&worker_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < num_allocs; j++) {
		page = alloc_pages(GFP_USER, alloc_config.order);
		BUG_ON(page == NULL);
		__free_pages(page, alloc_config.order);
	}
	timer = (ktime_get_ns() - timer) / num_allocs;
	t_perf->get = timer;
	t_perf->put = timer;
}

/// Random free and realloc
__maybe_unused static void rand(u64 num_allocs)
{
	u64 timer;
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct thread_perf *t_perf = this_cpu_ptr(&thread_perf);
	u64 threads = atomic64_read(&curr_threads);

	struct page **pages =
		kmalloc_array(num_allocs, sizeof(struct page *), GFP_KERNEL);
	BUG_ON(pages == NULL);

	for (u64 j = 0; j < num_allocs; j++) {
		pages[j] = alloc_pages(GFP_USER, alloc_config.order);
		BUG_ON(pages[j] == NULL);
	}
	rand_pages[raw_smp_processor_id()] = pages;

	// complete initialization
	complete(worker_complete);

	// Start allocations
	wait_for_completion(&worker_barrier0);

	// shuffle between all threads
	if (raw_smp_processor_id() == 0) {
		u64 rng = 42;
		pr_info("shuffle: a=%llu t=%llu\n", num_allocs, threads);
		for (u64 i = 0; i < num_allocs * threads; i++) {
			u64 j = nanorand_random_range(&rng, 0,
						      num_allocs * threads);
			swap(rand_pages[i % threads][i / threads],
			     rand_pages[j % threads][j / threads]);
		}
		pr_info("setup finished\n");
	}

	complete(worker_complete);

	// Start reallocs
	wait_for_completion(&worker_barrier);

	timer = ktime_get_ns();
	for (u64 j = 0; j < num_allocs; j++) {
		__free_pages(pages[j], alloc_config.order);
	}
	timer = (ktime_get_ns() - timer) / num_allocs;
	t_perf->get = timer;
	t_perf->put = timer;

	kfree(pages);
}

static int worker(void *data)
{
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct completion *worker_activate = this_cpu_ptr(&worker_activates);

	pr_info("Worker %u bench %u\n", smp_processor_id(), alloc_config.bench);

	for (;;) {
		wait_for_completion(worker_activate);
		if (kthread_should_stop() || !running) {
			pr_info("Stopping worker %d\n", smp_processor_id());
			return 0;
		}

		reinit_completion(worker_activate);

		if (alloc_config.bench == BENCH_BULK) {
			bulk(alloc_config.allocs);
		} else if (alloc_config.bench == BENCH_REPEAT) {
			repeat(alloc_config.allocs);
		} else if (alloc_config.bench == BENCH_RAND) {
			rand(alloc_config.allocs);
		}

		complete(worker_complete);
	}

	return 0;
}

/// Outputs the measured data.
/// Note: `buf` is PAGE_SIZE large!
static ssize_t out_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	struct perf *p;
	ssize_t len = 0;

	if (running || measurements == NULL)
		return -EINPROGRESS;

	if (out_index == 0) {
		len += sprintf(buf,
			       "alloc,x,iteration,allocs,get_min,get_avg,"
			       "get_max,put_min,put_avg,put_max,init,total\n");
	}

	for (ssize_t i = out_index; i < alloc_config.threads_len; i++) {
		// The output buffer has only the size of a PAGE.
		// If our output is larger we have to output it in multiple steps.
		if (len < PAGE_SIZE - alloc_config.iterations * 128) {
			for (ssize_t iter = 0; iter < alloc_config.iterations;
			     iter++) {
				p = &measurements[i * alloc_config.iterations +
						  iter];

				len += sprintf(
					buf + len,
					"Kernel,%llu,%lu,%llu,%llu,%llu,%llu,%llu,"
					"%llu,%llu,0,0\n",
					alloc_config.threads[i], iter,
					alloc_config.allocs, p->get_min,
					p->get_avg, p->get_max, p->put_min,
					p->put_avg, p->put_max);
			}
		} else {
			out_index = i;
			return len;
		}
	}
	out_index = 0;
	return len;
}

void iteration(u32 bench, u64 i, u64 iter)
{
	struct perf *p;
	u64 threads = alloc_config.threads[i];
	atomic64_set(&curr_threads, threads);

	pr_info("Start threads %llu\n", threads);
	for (u64 t = 0; t < threads; t++) {
		struct completion *worker_activate =
			per_cpu_ptr(&worker_activates, t);
		complete(worker_activate);
	}

	// Init
	for (u64 t = 0; t < threads; t++) {
		struct completion *worker_complete =
			per_cpu_ptr(&worker_completes, t);
		wait_for_completion(worker_complete);
		reinit_completion(worker_complete);
	}

	pr_info("Waiting for %llu workers...\n", threads);

	// bulk and rand have two phases (alloc and free)
	if (bench == BENCH_BULK || bench == BENCH_RAND) {
		complete_all(&worker_barrier0);
		for (u64 t = 0; t < threads; t++) {
			struct completion *worker_complete =
				per_cpu_ptr(&worker_completes, t);
			wait_for_completion(worker_complete);
			reinit_completion(worker_complete);
		}
		reinit_completion(&worker_barrier0);
	}

	complete_all(&worker_barrier);

	p = &measurements[i * alloc_config.iterations + iter];
	p->get_min = (u64)-1;
	p->get_avg = 0;
	p->get_max = 0;
	p->put_min = (u64)-1;
	p->put_avg = 0;
	p->put_max = 0;
	for (u64 t = 0; t < threads; t++) {
		struct completion *worker_complete =
			per_cpu_ptr(&worker_completes, t);
		struct thread_perf *t_perf = per_cpu_ptr(&thread_perf, t);
		u64 get, put;

		wait_for_completion(worker_complete);
		reinit_completion(worker_complete);

		get = t_perf->get;
		put = t_perf->put;

		p->get_min = min(p->get_min, get);
		p->get_avg += get;
		p->get_max = max(p->get_max, get);
		p->put_min = min(p->put_min, put);
		p->put_avg += put;
		p->put_max = max(p->put_max, put);
	}
	p->get_avg /= threads;
	p->put_avg /= threads;

	reinit_completion(&worker_barrier);

	pr_info("Finish iteration\n");
}

static bool whitespace(char c)
{
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static const char *str_skip(const char *buf, const char *end, bool ws)
{
	for (; buf < end && whitespace(*buf) == ws; buf++)
		;
	return buf;
}

static const char *next_uint(const char *buf, const char *end, u64 *dst)
{
	char *next;
	buf = str_skip(buf, end, true);

	if (buf >= end)
		return NULL;

	*dst = simple_strtoull(buf, &next, 10);
	if (next <= buf)
		return NULL;

	return next;
}

// just parsing a list of integers...
static const char *next_uint_list(const char *buf, const char *end, u64 **list,
				  u64 *list_len)
{
	u64 *threads;
	u64 threads_len = 1;
	char buffer[24];
	u64 n = 0;
	u64 bi = 0;
	// skip whitespace
	buf = str_skip(buf, end, true);
	if (buf >= end)
		return NULL;

	// count number of thread counts
	for (const char *tmp = buf; tmp < end && !whitespace(*tmp); tmp++) {
		if (*tmp == ',') {
			threads_len += 1;
		}
	}
	if (threads_len == 0)
		return NULL;

	// parse thread counts
	threads = kmalloc_array(threads_len, sizeof(u64), GFP_KERNEL);
	for (; buf < end && !whitespace(*buf); buf++) {
		if (*buf == ',') {
			if (bi == 0) {
				kfree(threads);
				return NULL;
			}
			buffer[bi] = '\0';
			if (kstrtou64(buffer, 10, &threads[n]) < 0) {
				kfree(threads);
				return NULL;
			}

			n += 1;
			bi = 0;
		} else {
			buffer[bi] = *buf;
			bi++;

			if (bi >= 24) {
				kfree(threads);
				return NULL;
			}
		}
	}

	if (bi == 0) {
		kfree(threads);
		return NULL;
	}
	buffer[bi] = '\0';
	if (kstrtou64(buffer, 10, &threads[n]) < 0) {
		kfree(threads);
		return NULL;
	}

	*list = threads;
	*list_len = threads_len;
	return buf;
}

/// Usage: <bench> <threads> <iterations> <allocs> <order>
static bool argparse(const char *buf, size_t len, struct alloc_config *args)
{
	u64 *threads;
	u64 threads_len;
	u64 iterations;
	u64 allocs;
	u64 order;
	const char *end = buf + len;

	if (len == 0 || buf == NULL || args == NULL) {
		pr_err("usage: <bench> <threads> <iterations> <allocs>");
		return false;
	}

	if (strncmp(buf, "bulk", min(len, 4ul)) == 0) {
		args->bench = BENCH_BULK;
		buf += 4;
	} else if (strncmp(buf, "repeat", min(len, 6ul)) == 0) {
		args->bench = BENCH_REPEAT;
		buf += 6;
	} else if (strncmp(buf, "rand", min(len, 4ul)) == 0) {
		args->bench = BENCH_RAND;
		buf += 4;
	} else {
		pr_err("Invalid mode %s", buf);
		return false;
	}

	if ((buf = next_uint_list(buf, end, &threads, &threads_len)) == NULL) {
		pr_err("Invalid <threads>");
		return false;
	}
	if ((buf = next_uint(buf, end, &iterations)) == NULL) {
		pr_err("Invalid <iterations>");
		return false;
	}
	if ((buf = next_uint(buf, end, &allocs)) == NULL) {
		pr_err("Invalid <allocs>");
		return false;
	}
	if ((buf = next_uint(buf, end, &order)) == NULL) {
		pr_err("Invalid <order>");
		return false;
	}

	buf = str_skip(buf, end, true);
	if (buf != end)
		return false;

	if (args->threads)
		kfree(args->threads);
	args->threads = threads;
	args->threads_len = threads_len;
	args->iterations = iterations;
	args->allocs = allocs;
	args->order = order;
	return true;
}

static ssize_t run_store(struct kobject *kobj, struct kobj_attribute *attr,
			 const char *buf, size_t len)
{
	u64 max_threads = 0;

	if (running)
		return -EINPROGRESS;

	if (!argparse(buf, len, &alloc_config))
		return -EINVAL;

	running = true;

	for (u64 i = 0; i < alloc_config.threads_len; i++) {
		max_threads = max(alloc_config.threads[i], max_threads);
	}

	if (measurements)
		kfree(measurements);

	measurements = kmalloc_array(alloc_config.threads_len *
					     alloc_config.iterations,
				     sizeof(struct perf), GFP_KERNEL);

	// Initialize workers in advance
	for (u64 t = 0; t < max_threads; t++) {
		struct completion *worker_commit =
			per_cpu_ptr(&worker_completes, t);
		struct completion *worker_wait =
			per_cpu_ptr(&worker_activates, t);
		struct task_struct **task = per_cpu_ptr(&per_cpu_tasks, t);
		init_completion(worker_commit);
		init_completion(worker_wait);
		*task = kthread_run_on_cpu(worker, NULL, t, "worker");
	}

	for (u64 i = 0; i < alloc_config.threads_len; i++) {
		for (u64 iter = 0; iter < alloc_config.iterations; iter++) {
			iteration(alloc_config.bench, i, iter);
		}
	}

	pr_info("Cleanup\n");

	running = false;
	for (u64 t = 0; t < max_threads; t++) {
		complete(per_cpu_ptr(&worker_activates, t));
	}

	pr_info("Finished\n");

	return len;
}

static struct kobj_attribute out_attribute = __ATTR(out, 0444, out_show, NULL);
static struct kobj_attribute run_attribute = __ATTR(run, 0220, NULL, run_store);

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

	rand_pages =
		kmalloc_array(num_present_cpus(), sizeof(void *), GFP_KERNEL);

	return 0;
}

static void alloc_cleanup_module(void)
{
	pr_info("End\n");
	kobject_put(output);
	if (alloc_config.threads)
		kfree(alloc_config.threads);
	if (measurements)
		kfree(measurements);
	kfree(rand_pages);
}

module_init(alloc_init_module);
module_exit(alloc_cleanup_module);
