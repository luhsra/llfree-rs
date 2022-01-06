use std::sync::atomic::Ordering;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use nvalloc::util::logging;
use nvalloc::{Alloc, Allocator, Page, Size, MIN_PAGES};

fn alloc_free_small_normal(c: &mut Criterion) {
    logging();

    let mut memory = vec![Page::new(); 8 * MIN_PAGES];
    c.bench_function("alloc_free_small_normal", |b| {
        b.iter(|| {
            Allocator::init(1, &mut memory).unwrap();

            let mut pages = Vec::with_capacity(MIN_PAGES);
            for _ in 0..MIN_PAGES {
                pages.push(nvalloc::get(black_box(0), Size::L0).unwrap());
            }
            pages = black_box(pages);
            for page in pages {
                nvalloc::put(black_box(0), page).unwrap();
            }

            Allocator::instance()
                .meta()
                .magic
                .store(0, Ordering::SeqCst);
            Allocator::uninit();
        });
    });
}

fn alloc_free_small_direct(c: &mut Criterion) {
    logging();

    let mut memory = vec![Page::new(); 8 * MIN_PAGES];
    c.bench_function("alloc_free_small_direct", |b| {
        b.iter(|| {
            Allocator::init(1, &mut memory).unwrap();

            for _ in 0..MIN_PAGES {
                let page = Allocator::get(black_box(0), Size::L0).unwrap();
                let page = black_box(page);
                Allocator::put(black_box(0), page).unwrap();
            }
            Allocator::instance()
                .meta()
                .magic
                .store(0, Ordering::SeqCst);
            Allocator::uninit();
        });
    });
}

criterion_group!(benches, alloc_free_small_normal, alloc_free_small_direct);
criterion_main!(benches);
