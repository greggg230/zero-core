//! Where a caught panic happened (docs/search.md 2.18): a dropped game's `GameFailure::site`.
//!
//! A panic's payload holds its message only; its location goes to the panic hook. `install` (once per process, by
//! `Driver::new`) puts a hook in front of the current one that notes the location (`file:line:col`) in a thread-local
//! and then calls the previous hook unchanged (the usual stderr line stays). The driver clears the note before a game's
//! step (`clear`) and takes it after catching that step's panic (`take`): the same thread, so the note is that panic's.
//! A hook installed later that does not chain to this one leaves the note empty: the failure's `site` is then None.
use std::cell::RefCell;
use std::sync::Once;

thread_local! {
    static SITE: RefCell<Option<String>> = const { RefCell::new(None) };
}

static INSTALL: Once = Once::new();

/// Put the noting hook in front of the current panic hook (once per process; later calls do nothing).
pub fn install() {
    INSTALL.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(l) = info.location() {
                let s = format!("{}:{}:{}", l.file(), l.line(), l.column());
                let _ = SITE.try_with(|c| c.try_borrow_mut().map(|mut b| *b = Some(s)));
            }
            prev(info);
        }));
    });
}

/// Forget this thread's last noted location.
pub fn clear() {
    let _ = SITE.try_with(|c| c.try_borrow_mut().map(|mut b| *b = None));
}

/// This thread's last noted location since `clear` (and forget it).
pub fn take() -> Option<String> {
    SITE.try_with(|c| c.try_borrow_mut().ok().and_then(|mut b| b.take())).ok().flatten()
}
