//! C ABI, opaque owned handles. All strings are UTF-8; outputs preserve tensor shape.
use crate::{Embedding, Encoder};
use std::{
    cell::RefCell,
    ffi::{c_char, CStr},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
};
thread_local! { static LAST: RefCell<String> = const { RefCell::new(String::new()) }; }
fn handle<T>(action: impl FnOnce() -> Result<T, crate::Error>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(value)) => {
            LAST.with(|e| e.borrow_mut().clear());
            Box::into_raw(Box::new(value))
        }
        Ok(Err(e)) => {
            LAST.with(|last| *last.borrow_mut() = e.to_string());
            ptr::null_mut()
        }
        Err(_) => {
            LAST.with(|last| *last.borrow_mut() = "native embedding panic".into());
            ptr::null_mut()
        }
    }
}
unsafe fn text<'a>(p: *const c_char) -> Result<&'a str, crate::Error> {
    if p.is_null() {
        return Err(crate::Error::artifact("NULL string"));
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|_| crate::Error::artifact("invalid UTF-8"))
}
#[no_mangle]
pub unsafe extern "C" fn ge_open(image: *const c_char, threads: usize) -> *mut Encoder {
    handle(|| {
        let options = if threads == 0 { None } else { Some(threads) };
        if image.is_null() {
            Encoder::open_image(
                std::env::current_exe().map_err(|e| crate::Error::io("locating executable", e))?,
                options,
            )
        } else {
            Encoder::open_image(text(image)?, options)
        }
    })
}
#[no_mangle]
pub unsafe extern "C" fn ge_close(encoder: *mut Encoder) {
    if !encoder.is_null() {
        drop(Box::from_raw(encoder));
    }
}
#[no_mangle]
pub unsafe extern "C" fn ge_embed_file(
    encoder: *const Encoder,
    path: *const c_char,
) -> *mut Embedding {
    handle(|| {
        encoder
            .as_ref()
            .ok_or_else(|| crate::Error::artifact("NULL encoder"))?
            .embed_file(text(path)?)
    })
}
#[no_mangle]
pub unsafe extern "C" fn ge_embed_pcm(
    encoder: *const Encoder,
    data: *const f32,
    count: usize,
    rate: u32,
) -> *mut Embedding {
    handle(|| {
        let enc = encoder
            .as_ref()
            .ok_or_else(|| crate::Error::artifact("NULL encoder"))?;
        if data.is_null() || count == 0 || count > isize::MAX as usize / std::mem::size_of::<f32>()
        {
            return Err(crate::Error::artifact("invalid PCM buffer"));
        }
        enc.embed_pcm(std::slice::from_raw_parts(data, count), rate)
    })
}
#[no_mangle]
pub unsafe extern "C" fn ge_embedding_free(value: *mut Embedding) {
    if !value.is_null() {
        drop(Box::from_raw(value));
    }
}
#[no_mangle]
pub unsafe extern "C" fn ge_embedding_data(
    value: *const Embedding,
    count: *mut usize,
) -> *const f32 {
    if let (Some(v), Some(n)) = (value.as_ref(), count.as_mut()) {
        *n = v.values.len();
        v.values.as_ptr()
    } else {
        ptr::null()
    }
}
#[no_mangle]
pub unsafe extern "C" fn ge_embedding_shape(
    value: *const Embedding,
    rank: *mut usize,
) -> *const usize {
    if let (Some(v), Some(n)) = (value.as_ref(), rank.as_mut()) {
        *n = v.shape.len();
        v.shape.as_ptr()
    } else {
        ptr::null()
    }
}
#[no_mangle]
pub unsafe extern "C" fn ge_embedding_mask(
    value: *const Embedding,
    count: *mut usize,
) -> *const bool {
    if let (Some(v), Some(n)) = (value.as_ref(), count.as_mut()) {
        *n = v.mask.len();
        v.mask.as_ptr()
    } else {
        ptr::null()
    }
}
unsafe fn copy_string(text: &str, out: *mut c_char, capacity: usize) -> usize {
    if !out.is_null() && capacity > 0 {
        let mut n = text.len().min(capacity - 1);
        while !text.is_char_boundary(n) {
            n -= 1;
        }
        ptr::copy_nonoverlapping(text.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
    }
    text.len() + 1
}
#[no_mangle]
pub unsafe extern "C" fn ge_last_error(out: *mut c_char, capacity: usize) -> usize {
    LAST.with(|e| copy_string(&e.borrow(), out, capacity))
}
#[no_mangle]
pub unsafe extern "C" fn ge_embedding_identity(
    value: *const Embedding,
    out: *mut c_char,
    capacity: usize,
) -> usize {
    value
        .as_ref()
        .map(|v| copy_string(&v.identity, out, capacity))
        .unwrap_or(0)
}
