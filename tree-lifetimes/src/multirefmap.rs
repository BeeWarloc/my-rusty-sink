use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;
use std::cell::RefCell;

// Just a straight forward ref to ref map, for any ref. 
pub struct RefCache<'a> {
    ref_map: RefCell<HashMap<usize, usize>>,
    phantom: PhantomData<&'a usize>
}

impl<'a> RefCache<'a> {
    fn insert<T: 'a>(&self, key: &'a T, value: &'a T) {
        unsafe {
            self.ref_map.borrow_mut().insert(mem::transmute::<&'a T, usize>(key), mem::transmute::<&'a T, usize>(value));
        }
    }

    fn get<T: 'a>(&self, key: &'a T) -> Option<&'a T> {
        unsafe {
            self.ref_map.borrow().get(&mem::transmute::<&'a T, usize>(key)).map(|ptr| mem::transmute::<usize, &'a T>(*ptr))
        }
    }

    pub fn get_or_insert<T: 'a, F: FnOnce() -> &'a T>(&self, key: &'a T, creator: F) -> &'a T {
        if let Some(val) = self.get(key) {
            return val;
        }

        let val = creator();
        self.insert(key, val);
        val
    }
}