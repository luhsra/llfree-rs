use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use crate::util::WyRand;
use log::{error, trace};

#[derive(Debug)]
pub enum Error {
    OverflowOrdering,
    WakeSuspended,
}
pub type Result<T> = std::result::Result<T, Error>;

struct StopData {
    tid: usize,
    inner: Arc<dyn Stop>,
}

thread_local! {
    /// Per thread wait data.
    static DATA: RefCell<Option<StopData>> = RefCell::new(None);
}

/// Manages per thread stopping behavior.
pub struct Stopper;

impl Stopper {
    /// Per thread initialization.
    /// Has to be called on every thread that calls wait.
    pub fn init(inner: Arc<dyn Stop>, tid: usize) -> Result<Self> {
        DATA.with(|data| {
            *data.borrow_mut() = Some(StopData { tid, inner });
            Ok(Stopper)
        })
    }
}

impl Drop for Stopper {
    fn drop(&mut self) {
        DATA.with(|data| {
            let mut data = data.borrow_mut();
            let data = data.take().unwrap();

            // Tell the others that we finished
            trace!("Finished {}", data.tid);
            data.inner.complete(data.tid).unwrap();
            trace!("Stop {}", data.tid);
        })
    }
}

/// Synchronization point with the other threads.
#[track_caller]
pub fn stop() -> Result<()> {
    let caller = core::panic::Location::caller();
    // Check if activated
    DATA.with(|data| {
        let mut data = data.borrow_mut();
        if let Some(data) = data.as_mut() {
            loop {
                trace!("Wait t{}", data.tid);

                if data.inner.stop(data.tid)? {
                    trace!("Run t{} {caller}", data.tid);
                    return Ok(());
                }
            }
        } else {
            Ok(())
        }
    })
}

/// Chooses the next threads to execute.
pub trait Stop {
    /// Waits and returns true if this thread may continue.
    fn stop(&self, tid: usize) -> Result<bool>;
    /// Waits for the other threads to complete.
    fn complete(&self, tid: usize) -> Result<()>;
}

/// Stops and continues threads base on predefined order.
pub struct StopVec {
    threads: usize,
    barrier: Barrier,
    order: Vec<usize>,
    i: AtomicUsize,
    finished: AtomicUsize,
}

impl StopVec {
    /// Initializes the order and number of threads.
    pub fn new(threads: usize, order: Vec<usize>) -> Arc<Self> {
        let mut finished = Vec::with_capacity(threads as _);
        finished.resize_with(threads as _, || AtomicBool::new(false));

        Arc::new(Self {
            threads,
            barrier: Barrier::new(threads),
            order,
            i: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
        })
    }
}
impl Stop for StopVec {
    fn stop(&self, tid: usize) -> Result<bool> {
        if self.barrier.wait().is_leader() {
            self.i.fetch_add(1, Ordering::SeqCst);
        }
        self.barrier.wait();
        let i = self.i.load(Ordering::SeqCst) - 1;

        if i >= self.order.len() {
            error!("Overflow ordering {i} >= {}", self.order.len());
            Err(Error::OverflowOrdering)
        } else {
            Ok(self.order[i] == tid)
        }
    }
    fn complete(&self, tid: usize) -> Result<()> {
        if self.barrier.wait().is_leader() {
            self.i.fetch_add(1, Ordering::SeqCst);
        }
        self.finished.fetch_add(1, Ordering::SeqCst);

        loop {
            self.barrier.wait();

            if self.finished.load(Ordering::SeqCst) == self.threads {
                return Ok(());
            }

            let i = self.i.load(Ordering::SeqCst) - 1;
            if i >= self.order.len() {
                error!("Overflow ordering for {tid}");
                return Err(Error::OverflowOrdering);
            }
            if self.order[i] == tid {
                error!("Wake suspended {tid} for {i}");
                return Err(Error::WakeSuspended);
            }

            if self.barrier.wait().is_leader() {
                self.i.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

/// Stops and continues threads base seeded rng.
pub struct StopRand {
    threads: usize,
    barrier: Barrier,
    seed: AtomicU64,
    finished: AtomicUsize,
}

impl StopRand {
    /// Initializes the order and number of threads. Has to be called before the init calls.
    pub fn new(threads: usize, seed: u64) -> Arc<Self> {
        Arc::new(Self {
            threads,
            barrier: Barrier::new(threads),
            seed: AtomicU64::new(seed),
            finished: AtomicUsize::new(0),
        })
    }
}
impl Stop for StopRand {
    fn stop(&self, tid: usize) -> Result<bool> {
        self.barrier.wait();

        let mut rng = WyRand::new(self.seed.load(Ordering::SeqCst));
        let v = rng.range(0..self.threads as _) as usize;

        if self.barrier.wait().is_leader() {
            self.seed.store(rng.seed, Ordering::SeqCst);
        }

        Ok(v == tid)
    }
    fn complete(&self, _tid: usize) -> Result<()> {
        self.barrier.wait();
        self.finished.fetch_add(1, Ordering::SeqCst);

        loop {
            if self.barrier.wait().is_leader() {
                let mut rng = WyRand::new(self.seed.load(Ordering::SeqCst));
                rng.gen();
                self.seed.store(rng.seed, Ordering::SeqCst);
            }

            if self.finished.load(Ordering::SeqCst) >= self.threads {
                return Ok(());
            }

            self.barrier.wait();
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use log::trace;

    use crate::util::logging;
    use crate::stop::{stop, Stopper, StopRand, StopVec};

    #[test]
    fn stop_vec() {
        logging();
        let b = StopVec::new(2, vec![0, 0, 1, 1, 0, 1]);

        let counter = Arc::new(AtomicUsize::new(0));

        let counter_c = counter.clone();

        let b1 = b.clone();
        let handle = thread::spawn(move || {
            let _key = Stopper::init(b1, 1).unwrap();

            stop().unwrap();
            trace!("thread 1: 0");
            counter_c
                .compare_exchange(2, 3, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            stop().unwrap();
            trace!("thread 1: 1");
            counter_c
                .compare_exchange(3, 4, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            stop().unwrap();
            trace!("thread 1: 2");
            counter_c
                .compare_exchange(5, 6, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();
        });

        {
            let _key = Stopper::init(b, 0).unwrap();

            stop().unwrap();
            trace!("thread 0: 0");
            counter
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            stop().unwrap();
            trace!("thread 0: 1");
            counter
                .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            stop().unwrap();
            trace!("thread 0: 2");
            counter
                .compare_exchange(4, 5, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();
        }

        handle.join().unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn wait_rand() {
        logging();
        let b = StopRand::new(2, 1);

        let counter = Arc::new(AtomicUsize::new(0));

        let counter_c = counter.clone();

        let b1 = b.clone();
        let handle = thread::spawn(move || {
            let _key = Stopper::init(b1, 1).unwrap();

            stop().unwrap();
            trace!("thread 1: 0");
            counter_c.fetch_add(1, Ordering::SeqCst);

            stop().unwrap();
            trace!("thread 1: 1");
            counter_c.fetch_add(1, Ordering::SeqCst);

            stop().unwrap();
            trace!("thread 1: 2");
            counter_c.fetch_add(1, Ordering::SeqCst);
        });

        {
            let _key = Stopper::init(b, 0).unwrap();

            stop().unwrap();
            trace!("thread 0: 0");
            counter.fetch_add(1, Ordering::SeqCst);

            stop().unwrap();
            trace!("thread 0: 1");
            counter.fetch_add(1, Ordering::SeqCst);

            stop().unwrap();
            trace!("thread 0: 2");
            counter.fetch_add(1, Ordering::SeqCst);
        }

        handle.join().unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 6);
    }
}
