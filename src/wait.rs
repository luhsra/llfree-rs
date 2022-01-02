#![cfg(feature = "wait")]

use std::cell::RefCell;
use std::sync::{Arc, Barrier};

use log::{error, trace};

pub struct DbgWait {
    barrier: Barrier,
    order: Vec<u8>,
}

unsafe impl Send for DbgWait {}

impl DbgWait {
    /// Initializes the order and number of threads. Has to be called before the init calls.
    pub fn setup(threads: u8, order: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            barrier: Barrier::new(threads as _),
            order,
        })
    }
}

struct WaitData {
    id: u8,
    index: usize,
    wait: Arc<DbgWait>,
}

#[derive(Debug)]
pub enum Error {
    OverflowOrdering,
}
pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    static DATA: RefCell<Option<WaitData>> = RefCell::new(None);
}

pub struct DbgWaitKey;

impl DbgWaitKey {
    /// Per thread initialization. Has to be called on every thread that calls wait.
    pub fn init(wait: Arc<DbgWait>, id: u8) -> Result<Self> {
        DATA.with(|data| {
            let mut data = data.borrow_mut();

            *data = Some(WaitData { id, index: 0, wait });
            Ok(DbgWaitKey)
        })
    }
}

impl Drop for DbgWaitKey {
    /// Has to be called at the end of a thread.
    fn drop(&mut self) {
        DATA.with(|data| {
            let mut data = data.borrow_mut();
            let data = data.take().unwrap();

            for i in data.index..data.wait.order.len() {
                trace!("t{} wait for {}", data.id, i);
                data.wait.barrier.wait();

                if data.wait.order[i] == data.id {
                    panic!("Wake suspended t{} for {}", data.id, i);
                }
            }
        })
    }
}

/// Synchronization point with the other threads.
pub fn wait() -> Result<()> {
    // Check if activated
    DATA.with(|data| {
        let mut data = data.borrow_mut();
        if let Some(data) = data.as_mut() {
            for i in data.index..data.wait.order.len() {
                trace!("t{} wait for {}", data.id, i);
                data.wait.barrier.wait();

                if data.wait.order[i] == data.id {
                    trace!("run t{} for {}", data.id, i);
                    data.index = i + 1;
                    return Ok(());
                }
            }

            error!("Sync overflow ordering {}", data.wait.order.len());
            Err(Error::OverflowOrdering)
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use log::trace;

    use crate::util::logging;
    use crate::wait::{wait, DbgWait, DbgWaitKey};

    #[test]
    fn sync() {
        logging();
        let b = DbgWait::setup(2, vec![0, 0, 1, 1, 0, 1]);

        let counter = Arc::new(AtomicUsize::new(0));

        let counter_c = counter.clone();

        let b1 = b.clone();
        let handle = thread::spawn(move || {
            let _key = DbgWaitKey::init(b1, 1).unwrap();

            wait().unwrap();
            trace!("thread 1: 0");
            counter_c
                .compare_exchange(2, 3, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            wait().unwrap();
            trace!("thread 1: 1");
            counter_c
                .compare_exchange(3, 4, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            wait().unwrap();
            trace!("thread 1: 2");
            counter_c
                .compare_exchange(5, 6, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();
        });

        {
            let _key = DbgWaitKey::init(b, 0).unwrap();

            wait().unwrap();
            trace!("thread 0: 0");
            counter
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            wait().unwrap();
            trace!("thread 0: 1");
            counter
                .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            wait().unwrap();
            trace!("thread 0: 2");
            counter
                .compare_exchange(4, 5, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();
        }

        handle.join().unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 6);
    }
}
