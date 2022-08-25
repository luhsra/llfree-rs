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

static __read_mostly u64 num_allocs = 512 * 512;
static __read_mostly u64 num_iterations = 4;
static __read_mostly u64 *num_threads = NULL;
static __read_mostly u64 num_threads_len = 0;

static DEFINE_PER_CPU(struct task_struct *, per_cpu_tasks);

static DEFINE_PER_CPU(struct completion, worker_completes);
static DEFINE_PER_CPU(struct completion, worker_activates);

static DECLARE_COMPLETION(worker_barrier);
static DECLARE_COMPLETION(worker_barrier0);

static bool running = false;

struct thread_perf {
	atomic64_t get;
	atomic64_t put;
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
static struct perf *measurements;
static u64 out_index = 0;

__maybe_unused static u64 cycles(void)
{
	u32 lo, hi;
	asm volatile("rdtsc" : "=eax"(lo), "=edx"(hi) :);
	return ((u64)lo) | ((u64)hi << 32);
};

/// Alloc a number of pages at once and free them afterwards
__maybe_unused static void bulk()
{
	u64 j;
	u64 timer;
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct thread_perf *t_perf = this_cpu_ptr(&thread_perf);
	struct page **pages =
		kmalloc_array(num_allocs, sizeof(struct page *), GFP_KERNEL);

	if (pages == NULL) {
		pr_err("kmalloc failed");
		return;
	}

	// complete initialization
	complete(worker_complete);

	// Start allocations
	wait_for_completion(&worker_barrier0);

	timer = ktime_get_ns();
	for (j = 0; j < num_allocs; j++) {
		pages[j] = alloc_page(GFP_USER);
		if (pages == NULL) {
			pr_err("alloc_page failed");
		}
	}
	timer = (ktime_get_ns() - timer) / num_allocs;
	atomic64_set(&t_perf->get, timer);

	complete(worker_complete);

	// Start frees
	wait_for_completion(&worker_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < num_allocs; j++) {
		__free_page(pages[j]);
	}
	timer = (ktime_get_ns() - timer) / num_allocs;
	atomic64_set(&t_perf->put, timer);
	kfree(pages);
}

/// Alloc and free the same page
__maybe_unused static void repeat()
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
		page = alloc_page(GFP_USER);
		if (page == NULL) {
			pr_err("alloc_page failed");
		}
		__free_page(page);
	}
	timer = (ktime_get_ns() - timer) / num_allocs;
	atomic64_set(&t_perf->get, timer);
	atomic64_set(&t_perf->put, timer);
}

/// Random free and realloc
__maybe_unused static void rand()
{
	u64 i, j;
	u64 timer;
	u64 rng = raw_smp_processor_id();
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct thread_perf *t_perf = this_cpu_ptr(&thread_perf);

	struct page **pages =
		kmalloc_array(num_allocs, sizeof(struct page *), GFP_KERNEL);
	if (pages == NULL) {
		pr_err("kmalloc failed");
		return;
	}

	for (j = 0; j < num_allocs; j++) {
		pages[j] = alloc_page(GFP_USER);
		if (pages == NULL) {
			pr_err("alloc_page failed");
		}
	}

	// complete initialization
	complete(worker_complete);

	// Start reallocs
	wait_for_completion(&worker_barrier);

	timer = ktime_get_ns();
	for (j = 0; j < num_allocs; j++) {
		i = nanorand_random_range(&rng, 0, num_allocs);

		__free_page(pages[i]);
		pages[i] = alloc_page(GFP_USER);
		if (pages[i] == NULL) {
			pr_err("alloc_page failed");
		}
	}
	timer = (ktime_get_ns() - timer) / num_allocs;
	atomic64_set(&t_perf->get, timer);
	atomic64_set(&t_perf->put, timer);

	for (j = 0; j < num_allocs; j++) {
		__free_page(pages[j]);
	}

	kfree(pages);
}

static int worker(void *data)
{
	u64 bench = (u64)data;
	struct completion *worker_complete = this_cpu_ptr(&worker_completes);
	struct completion *worker_activate = this_cpu_ptr(&worker_activates);

	pr_info("Worker %u bench %u\n", smp_processor_id(), bench);

	for (;;) {
		wait_for_completion(worker_activate);
		if (kthread_should_stop() || !running) {
			pr_info("Stopping worker %d", smp_processor_id());
			return 0;
		}

		reinit_completion(worker_activate);

		if (bench == 0) {
			bulk();
		} else if (bench == 1) {
			repeat();
		} else if (bench == 2) {
			rand();
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

	if (running)
		return -EINPROGRESS;

	if (out_index == 0) {
		len += sprintf(buf,
			       "alloc,x,iteration,allocs,get_min,get_avg,"
			       "get_max,put_min,put_avg,put_max,init,total\n");
	}

	for (ssize_t i = out_index; i < num_threads_len; i++) {
		// The output buffer has only the size of a PAGE.
		// If our output is larger we have to output it in multiple steps.
		if (len < PAGE_SIZE - num_iterations * 128) {
			for (ssize_t iter = 0; iter < num_iterations; iter++) {
				p = &measurements[i * num_iterations + iter];

				len += sprintf(
					buf + len,
					"Kernel,%llu,%lu,%llu,%llu,%llu,%llu,%llu,"
					"%llu,%llu,0,0\n",
					num_threads[i], iter, num_allocs,
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

ssize_t iteration(u32 bench, u64 i, u64 iter)
{
	struct perf *p;
	u64 get, put;
	u64 threads = num_threads[i];

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

	pr_info("Run on %llu threads\n", threads);
	p = &measurements[i * num_iterations + iter];
	p->get_min = (u64)-1;
	p->get_avg = 0;
	p->get_max = 0;
	p->put_min = (u64)-1;
	p->put_avg = 0;
	p->put_max = 0;

	pr_info("Waiting for workers...\n");

	// bulk alloc has two phases (alloc and free)
	if (bench == 0) {
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

	for (u64 t = 0; t < threads; t++) {
		struct completion *worker_complete =
			per_cpu_ptr(&worker_completes, t);
		struct thread_perf *t_perf = per_cpu_ptr(&thread_perf, t);

		wait_for_completion(worker_complete);
		reinit_completion(worker_complete);

		get = atomic64_read(&t_perf->get);
		put = atomic64_read(&t_perf->put);

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

	pr_info("Finish iteration");

	return 0;
}

static ssize_t run_store(struct kobject *kobj, struct kobj_attribute *attr,
			 const char *buf, size_t len)
{
	u64 bench = 0;

	if (len == 0 || len > 10 || buf == NULL)
		return -EINVAL;

	if (strncmp(buf, "bulk", 1) == 0) {
		bench = 0;
	} else if (strncmp(buf, "repeat", 1) == 0) {
		bench = 1;
	} else if (strncmp(buf, "rand", 1) == 0) {
		bench = 2;
	} else {
		pr_err("Invalid mode %s", buf);
		return -EINVAL;
	}

	if (running)
		return -EINPROGRESS;
	running = true;

	measurements = kmalloc_array(num_threads_len * num_iterations,
				     sizeof(struct perf), GFP_KERNEL);

	// Initialize workers in advance
	for (u64 t = 0; t < num_possible_cpus(); t++) {
		struct completion *worker_commit =
			per_cpu_ptr(&worker_completes, t);
		struct completion *worker_wait =
			per_cpu_ptr(&worker_activates, t);
		struct task_struct **task = per_cpu_ptr(&per_cpu_tasks, t);
		init_completion(worker_commit);
		init_completion(worker_wait);
		*task = kthread_run_on_cpu(worker, (void *)bench, t, "worker");
	}

	for (u64 i = 0; i < num_threads_len; i++) {
		for (u64 iter = 0; iter < num_iterations; iter++) {
			ssize_t ret = iteration(bench, i, iter);
			if (ret != 0) {
				running = false;
				return ret;
			}
		}
	}

	pr_info("Cleanup\n");
	kfree(measurements);

	running = false;
	for (u64 t = 0; t < num_possible_cpus(); t++) {
		complete(per_cpu_ptr(&worker_activates, t));
	}

	pr_info("Finished\n");

	return len;
}

static ssize_t allocs_show(struct kobject *kobj, struct kobj_attribute *attr,
			   char *buf)
{
	return sysfs_emit(buf, "%lu\n", num_allocs);
}

static ssize_t allocs_store(struct kobject *kobj, struct kobj_attribute *attr,
			    const char *buf, size_t len)
{
	int ret;
	if (running)
		return -EINPROGRESS;

	ret = kstrtou64(buf, 10, &num_allocs);
	if (ret < 0)
		return ret;
	return len;
}

static ssize_t iterations_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", num_iterations);
}

static ssize_t iterations_store(struct kobject *kobj,
				struct kobj_attribute *attr, const char *buf,
				size_t len)
{
	int ret;
	if (running)
		return -EINPROGRESS;

	ret = kstrtou64(buf, 10, &num_iterations);
	if (ret < 0)
		return ret;

	return len;
}

static ssize_t threads_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	ssize_t at;
	ssize_t r = sysfs_emit(buf, "%llu", num_threads[0]);
	if (r < 0)
		return r;
	at = r;
	for (size_t i = 1; i < num_threads_len; ++i) {
		r = sysfs_emit_at(buf, at, ",%llu", num_threads[i]);
		if (r < 0)
			return r;
		at += r;
	}
	r = sysfs_emit_at(buf, at, "\n");
	if (r < 0)
		return r;
	return at + r;
}

static ssize_t threads_store(struct kobject *kobj, struct kobj_attribute *attr,
			     const char *buf, size_t len)
{
	u64 *threads;
	u64 threads_len = 1;
	char buffer[24];
	int ret;
	u64 n = 0;
	u64 bi = 0;

	if (running)
		return -EINPROGRESS;

	// count number of thread counts
	for (size_t i = 0; i < len; i++) {
		if (buf[i] == ',') {
			threads_len += 1;
		}
	}

	// parse thread counts
	threads = kmalloc_array(threads_len, sizeof(u64), GFP_KERNEL);
	for (size_t i = 0; i < len; i++) {
		if (buf[i] == ',') {
			if (bi == 0) {
				kfree(threads);
				return -EINVAL;
			}
			buffer[bi] = '\0';
			ret = kstrtou64(buffer, 10, &threads[n]);
			if (ret < 0) {
				kfree(threads);
				return ret;
			}

			n += 1;
			bi = 0;
		} else {
			buffer[bi] = buf[i];
			bi++;

			if (bi >= 24) {
				kfree(threads);
				return -EINVAL;
			}
		}
	}

	if (bi == 0) {
		kfree(threads);
		return -EINVAL;
	}
	buffer[bi] = '\0';
	ret = kstrtou64(buffer, 10, &threads[n]);
	if (ret < 0) {
		kfree(threads);
		return ret;
	}

	// update parameters
	kfree(num_threads);
	num_threads = threads;
	num_threads_len = threads_len;

	return len;
}

static struct kobj_attribute out_attribute = __ATTR(out, 0444, out_show, NULL);
static struct kobj_attribute run_attribute = __ATTR(run, 0220, NULL, run_store);
static struct kobj_attribute allocs_attribute =
	__ATTR(allocs, 0664, allocs_show, allocs_store);
static struct kobj_attribute iterations_attribute =
	__ATTR(iterations, 0664, iterations_show, iterations_store);
static struct kobj_attribute threads_attribute =
	__ATTR(threads, 0664, threads_show, threads_store);

static struct attribute *attrs[] = {
	&out_attribute.attr,
	&run_attribute.attr,
	&allocs_attribute.attr,
	&iterations_attribute.attr,
	&threads_attribute.attr,
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

	// Init with 1
	num_threads = kmalloc_array(1, sizeof(u64), GFP_KERNEL);
	num_threads[0] = 1;
	num_threads_len = 1;

	return 0;
}

static void alloc_cleanup_module(void)
{
	pr_info("End\n");
	kobject_put(output);
	kfree(num_threads);
}

module_init(alloc_init_module);
module_exit(alloc_cleanup_module);
