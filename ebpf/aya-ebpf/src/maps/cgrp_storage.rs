use core::marker::PhantomData;

use aya_ebpf_bindings::bindings::bpf_map_type::BPF_MAP_TYPE_CGRP_STORAGE;

use crate::maps::{MapDef, PinningType};

/// A cgroup local storage map.
///
/// Cgroup storage maps store per-cgroup data. In eBPF programs, use
/// [`bpf_cgrp_storage_get`](aya_ebpf_bindings::helpers::bpf_cgrp_storage_get)
/// to access the data for the current cgroup.
///
/// # Minimum kernel version
///
/// Requires kernel 6.2 or later.
#[repr(transparent)]
pub struct CgrpStorage<V> {
    def: MapDef,
    _v: PhantomData<V>,
}

impl<V> CgrpStorage<V> {
    map_constructors!(
        i32, // key type: cgroup fd (in userspace) / cgroup pointer (in kernel)
        V,
        BPF_MAP_TYPE_CGRP_STORAGE,
        phantom _v,
    );

    /// Get a reference to the storage for the given cgroup.
    ///
    /// # Safety
    ///
    /// The `cgroup` pointer must be valid (e.g., from a BPF program context or helper return
    /// value). The returned reference borrows kernel-owned map data; the caller must ensure the
    /// cgroup outlives the reference.
    #[inline]
    pub unsafe fn get(&self, cgroup: *mut aya_ebpf_bindings::bindings::cgroup) -> Option<&V> {
        unsafe { self.get_ptr_impl(cgroup).map(|p| &*p) }
    }

    /// Get a mutable reference to the storage for the given cgroup.
    ///
    /// # Safety
    ///
    /// The `cgroup` pointer must be valid. The caller must ensure no other references to this
    /// storage value exist and that the cgroup outlives the reference.
    #[expect(
        clippy::mut_from_ref,
        reason = "BPF local storage returns kernel-owned memory; interior mutability is inherent"
    )]
    #[inline]
    pub unsafe fn get_mut(
        &self,
        cgroup: *mut aya_ebpf_bindings::bindings::cgroup,
    ) -> Option<&mut V> {
        unsafe { self.get_ptr_impl(cgroup).map(|p| &mut *p) }
    }

    /// Get a raw const pointer to the storage for the given cgroup.
    ///
    /// # Safety
    ///
    /// The `cgroup` pointer must be valid and point to a kernel cgroup struct.
    #[inline]
    pub unsafe fn get_ptr(
        &self,
        cgroup: *mut aya_ebpf_bindings::bindings::cgroup,
    ) -> Option<*const V> {
        unsafe { self.get_ptr_impl(cgroup).map(<*mut V>::cast_const) }
    }

    /// Get a raw mutable pointer to the storage for the given cgroup.
    ///
    /// # Safety
    ///
    /// The `cgroup` pointer must be valid and point to a kernel cgroup struct.
    #[inline]
    pub unsafe fn get_ptr_mut(
        &self,
        cgroup: *mut aya_ebpf_bindings::bindings::cgroup,
    ) -> Option<*mut V> {
        unsafe { self.get_ptr_impl(cgroup) }
    }

    /// Internal helper: calls `bpf_cgrp_storage_get` with flags=0 (lookup only).
    #[inline]
    unsafe fn get_ptr_impl(
        &self,
        cgroup: *mut aya_ebpf_bindings::bindings::cgroup,
    ) -> Option<*mut V> {
        let ptr = unsafe {
            aya_ebpf_bindings::helpers::bpf_cgrp_storage_get(
                self.def.as_ptr(),
                cgroup,
                core::ptr::null_mut(),
                0,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(ptr.cast::<V>())
        }
    }

    /// Get or initialize the storage for the given cgroup.
    ///
    /// If storage doesn't exist yet for this cgroup, it will be created and initialized with the
    /// provided `init_val`.
    ///
    /// # Safety
    ///
    /// The `cgroup` pointer must be valid. The returned reference borrows kernel-owned map data.
    #[expect(
        clippy::mut_from_ref,
        reason = "BPF local storage returns kernel-owned memory; interior mutability is inherent"
    )]
    #[inline]
    pub unsafe fn get_or_init(
        &self,
        cgroup: *mut aya_ebpf_bindings::bindings::cgroup,
        init_val: &V,
    ) -> Option<&mut V> {
        let ptr = unsafe {
            aya_ebpf_bindings::helpers::bpf_cgrp_storage_get(
                self.def.as_ptr(),
                cgroup,
                core::ptr::from_ref(init_val).cast_mut().cast(),
                // BPF_LOCAL_STORAGE_GET_F_CREATE = 1
                1,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &mut *ptr.cast::<V>() })
        }
    }

    /// Delete the storage for the given cgroup.
    ///
    /// # Safety
    ///
    /// The `cgroup` pointer must be valid.
    #[inline]
    pub unsafe fn delete(
        &self,
        cgroup: *mut aya_ebpf_bindings::bindings::cgroup,
    ) -> Result<(), i64> {
        let ret = unsafe {
            aya_ebpf_bindings::helpers::bpf_cgrp_storage_delete(self.def.as_ptr(), cgroup)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(ret)
        }
    }
}

unsafe impl<V: Sync> Sync for CgrpStorage<V> {}
