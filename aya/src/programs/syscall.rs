//! Syscall programs.

use std::os::fd::AsFd as _;

use aya_obj::generated::bpf_prog_type::BPF_PROG_TYPE_SYSCALL;

use crate::programs::{
    FdLink, FdLinkId, ProgramData, ProgramError, ProgramType, define_link_wrapper, load_program,
};
use crate::sys::bpf_prog_test_run;

/// A program that can be triggered via `BPF_PROG_TEST_RUN`.
///
/// Syscall programs are inherently sleepable and can call kfuncs that
/// require a sleepable context (such as `bpf_arena_alloc_pages`).
/// They are typically used for testing or for performing privileged
/// operations from user space via `BPF_PROG_RUN`.
///
/// # Minimum kernel version
///
/// The minimum kernel version required to use this feature is 5.14.
///
/// # Examples
///
/// ```no_run
/// # #[derive(Debug, thiserror::Error)]
/// # enum Error {
/// #     #[error(transparent)]
/// #     Program(#[from] aya::programs::ProgramError),
/// #     #[error(transparent)]
/// #     Ebpf(#[from] aya::EbpfError)
/// # }
/// # let mut bpf = aya::Ebpf::load(&[])?;
/// use aya::programs::SyscallProgram;
///
/// let prog: &mut SyscallProgram = bpf.program_mut("my_syscall_prog").unwrap().try_into()?;
/// prog.load()?;
/// prog.test_run()?;
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug)]
#[doc(alias = "BPF_PROG_TYPE_SYSCALL")]
pub struct SyscallProgram {
    pub(crate) data: ProgramData<SyscallProgramLink>,
}

impl SyscallProgram {
    /// The type of the program according to the kernel.
    pub const PROGRAM_TYPE: ProgramType = ProgramType::Syscall;

    /// Loads the program inside the kernel.
    pub fn load(&mut self) -> Result<(), ProgramError> {
        load_program(BPF_PROG_TYPE_SYSCALL, &mut self.data)
    }

    /// Runs the program once via `BPF_PROG_TEST_RUN` and returns its return value.
    pub fn test_run(&self) -> Result<u32, ProgramError> {
        let fd = self.data.fd()?;
        bpf_prog_test_run(fd.as_fd()).map_err(|io_error| {
            ProgramError::SyscallError(crate::sys::SyscallError {
                call: "bpf_prog_test_run",
                io_error,
            })
        })
    }
}

define_link_wrapper!(
    SyscallProgramLink,
    SyscallProgramLinkId,
    FdLink,
    FdLinkId,
    SyscallProgram
);
