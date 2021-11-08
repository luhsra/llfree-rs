use std::cell::RefCell;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Barrier;

use log::{error, info};

struct SyncData {
    id: u8,
    index: usize,
    barrier: *mut Barrier,
}

#[derive(Debug)]
pub enum Error {
    Uninitialized,
    WakeSuspended,
    OverflowOrdering,
}
pub type Result<T> = std::result::Result<T, Error>;

static mut ORDER: Vec<u8> = Vec::new();
static BARRIER: AtomicPtr<Barrier> = AtomicPtr::new(null_mut());

thread_local! {
    static DATA: RefCell<Option<SyncData>> = RefCell::new(None);
}

/// Initializes the order and number of threads. Has to be called before the init calls.
pub fn setup(threads: u8, order: Vec<u8>) {
    unsafe {
        ORDER = order;
        BARRIER.store(
            Box::leak(Box::new(Barrier::new(threads as _))),
            Ordering::SeqCst,
        );
    }
}

/// Per thread initialization. Has to be called on every thread that calls wait.
pub fn init(id: u8) -> Result<()> {
    DATA.with(|data| {
        let mut data = data.borrow_mut();
        let barrier = BARRIER.load(Ordering::SeqCst);
        if barrier.is_null() {
            return Err(Error::Uninitialized);
        }

        *data = Some(SyncData {
            id,
            index: 0,
            barrier,
        });
        Ok(())
    })
}

/// Synchronization point with the other threads.
pub fn wait() -> Result<()> {
    // Check if activated
    if BARRIER.load(Ordering::SeqCst).is_null() {
        return Ok(());
    }
    DATA.with(|data| {
        let mut data = data.borrow_mut();
        let data = data.as_mut().ok_or(Error::Uninitialized)?;

        for i in data.index..unsafe { ORDER.len() } {
            info!("t{} wait for {}", data.id, i);
            unsafe { &*data.barrier }.wait();

            if unsafe { ORDER[i] == data.id } {
                info!("run t{} for {}", data.id, i);
                data.index = i + 1;
                return Ok(());
            }
        }

        error!("Sync overflow ordering {}", unsafe { ORDER.len() });
        Err(Error::OverflowOrdering)
    })
}

/// Has to be called at the end of a thread.
pub fn end() -> Result<()> {
    // Check if activated
    if BARRIER.load(Ordering::SeqCst).is_null() {
        return Ok(());
    }
    DATA.with(|data| {
        let mut data = data.borrow_mut();
        let data = data.as_mut().ok_or(Error::Uninitialized)?;

        for i in data.index..unsafe { ORDER.len() } {
            info!("t{} wait for {}", data.id, i);
            unsafe { &*data.barrier }.wait();

            if unsafe { ORDER[i] == data.id } {
                error!("Wake suspended t{} for {}", data.id, i);
                return Err(Error::WakeSuspended);
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod test {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        thread,
    };

    use crate::util::logging;

    #[test]
    fn sync() {
        logging();
        super::setup(2, vec![0, 0, 1, 1, 0, 1]);

        let counter = Arc::new(AtomicUsize::new(0));

        let counter_c = counter.clone();
        let handle = thread::spawn(move || {
            super::init(1).unwrap();

            super::wait().unwrap();
            println!("thread 1: 0");
            counter_c
                .compare_exchange(2, 3, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            super::wait().unwrap();
            println!("thread 1: 1");
            counter_c
                .compare_exchange(3, 4, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            super::wait().unwrap();
            println!("thread 1: 2");
            counter_c
                .compare_exchange(5, 6, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap();

            super::end().unwrap();
        });
        super::init(0).unwrap();

        super::wait().unwrap();
        println!("thread 0: 0");
        counter
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap();

        super::wait().unwrap();
        println!("thread 0: 1");
        counter
            .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap();

        super::wait().unwrap();
        println!("thread 0: 2");
        counter
            .compare_exchange(4, 5, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap();

        super::end().unwrap();
        handle.join().unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 6);
    }
}
