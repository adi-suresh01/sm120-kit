use std::time::Duration;

use jiff::Timestamp;

/// Wall time and sleeping go through this trait so lifecycle tests can run on a fake clock without waiting.
pub trait Clock {
    fn now(&self) -> Timestamp;
    fn sleep(&self, duration: Duration);
}

pub struct SystemClock;

impl Clock for SystemClock {
    /// Whole seconds keep the ledger readable and make the pod's deadline epoch equal the recorded deadline.
    fn now(&self) -> Timestamp {
        let now = Timestamp::now();
        Timestamp::from_second(now.as_second()).unwrap_or(now)
    }

    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}
