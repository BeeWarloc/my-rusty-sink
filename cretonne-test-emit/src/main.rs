extern crate cretonne;
extern crate cton_frontend;
extern crate libc;
extern crate winapi;
extern crate kernel32;

use cretonne::entity::EntityRef;
use cretonne::ir::{self, ExternalName, CallConv, Function, Signature, AbiParam, InstBuilder, JumpTable};
use cretonne::ir::types::*;
use cretonne::settings::{self, Configurable};
use cton_frontend::{ILBuilder, FunctionBuilder};
use cretonne::verifier::verify_function;
use cretonne::isa;
use cretonne::binemit::{self, MemoryCodeSink, RelocSink, Reloc, CodeOffset, Addend};
use std::u32;
use std::fs;
use std::io::{Write, BufWriter};
use std::fmt::Write as FmtWrite;

use jitter::*;

fn main() {
    let mut sig = Signature::new(CallConv::Native);
    sig.returns.push(AbiParam::new(F64));
    sig.params.push(AbiParam::new(F64));
    sig.params.push(AbiParam::new(F64));
    let mut il_builder = ILBuilder::<Variable>::new();
    let mut func = Function::with_name_signature(ExternalName::user(0, 0), sig);
    {
        let mut builder = FunctionBuilder::<Variable>::new(&mut func, &mut il_builder);
        let block0 = builder.create_ebb();
        let x = Variable(0);
        builder.declare_var(x, F64);
        builder.append_ebb_params_for_function_params(block0);
        builder.switch_to_block(block0);
        {
            let a = builder.ebb_params(block0)[0];
            let b = builder.ebb_params(block0)[1];
            let tmp = builder.ins().fmul(a, b);
            let tmp = builder.ins().fadd(a, tmp);
            let tmp = builder.ins().sqrt(tmp);
            builder.ins().return_(&[tmp]);
        }
        builder.seal_block(block0);
        builder.finalize();
    }

    let mut shared_builder = settings::builder();
    shared_builder.set("is_64bit", "1").expect("Can't set flag");
    let flags = settings::Flags::new(&shared_builder);
    let res = verify_function(&func, &flags).unwrap();
    let arch_name = "intel";
    let mut isa_builder = isa::lookup(arch_name).expect("Arch not found");
    let mut reloc_sink = NopRelocSink {};
    let x64isa = isa_builder.finish(flags);
    let isa: &isa::TargetIsa = x64isa.as_ref();
    let mut buffer = [0u8; 1024 * 64];
    {
        let p = unsafe { &mut buffer[0] as *mut u8 };
        let mut comp_ctx = cretonne::Context::for_function(func);
        println!("{}", comp_ctx.func.display(None));

        comp_ctx.compile(isa).expect("Compile failed");
        println!("{}", comp_ctx.func.display(isa));
        let mut sink = MemoryCodeSink::new(p, &mut reloc_sink);
        isa.emit_function(&comp_ctx.func, &mut sink);
        // TODO: need to cast p to: extern "C" fn(f64, f64) -> f64
        // https://users.rust-lang.org/t/function-pointers-and-raw-function-pointers/15152/14
        // http://www.jonathanturner.org/2015/12/building-a-simple-jit-in-rust.html

        let fp = unsafe { std::mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(p) };
        //println!("result: {}", fp(20.0, 30.0));
    }

    let mut jitter = Jitter::new(&buffer);
    println!("{} {}", jitter.run_f64_f64_f64(20.0, 30.0),  (((20.0 * 30.0) + 20.0) as f64).sqrt());

    let f = fs::File::create("code").expect("Unable to create file");
    let mut f = BufWriter::new(f);
    f.write_all(&buffer).unwrap();

    println!("Hello, world!");
}


// Code sink that generates text.
struct TextSink {
    offset: binemit::CodeOffset,
    text: String,
}

impl TextSink {
    /// Create a new empty TextSink.
    pub fn new() -> Self {
        Self {
            offset: 0,
            text: String::new(),
        }
    }
}



impl binemit::CodeSink for TextSink {
    fn offset(&self) -> binemit::CodeOffset {
        self.offset
    }

    fn put1(&mut self, x: u8) {
        write!(self.text, "{:02x} ", x).unwrap();
        self.offset += 1;
    }

    fn put2(&mut self, x: u16) {
        write!(self.text, "{:04x} ", x).unwrap();
        self.offset += 2;
    }

    fn put4(&mut self, x: u32) {
        write!(self.text, "{:08x} ", x).unwrap();
        self.offset += 4;
    }

    fn put8(&mut self, x: u64) {
        write!(self.text, "{:016x} ", x).unwrap();
        self.offset += 8;
    }

    fn reloc_ebb(&mut self, reloc: binemit::Reloc, ebb_offset: binemit::CodeOffset) {
        write!(self.text, "{}({}) ", reloc, ebb_offset).unwrap();
    }

    fn reloc_external(
        &mut self,
        reloc: binemit::Reloc,
        name: &ir::ExternalName,
        addend: binemit::Addend,
    ) {
        write!(
            self.text,
            "{}({}",
            reloc,
            name,
        ).unwrap();
        if addend != 0 {
            write!(
                self.text,
                "{:+}",
                addend,
            ).unwrap();
        }
        write!(
            self.text,
            ") ",
        ).unwrap();
    }

    fn reloc_jt(&mut self, reloc: binemit::Reloc, jt: ir::JumpTable) {
        write!(self.text, "{}({}) ", reloc, jt).unwrap();
    }
}


pub struct NopRelocSink {}
impl RelocSink for NopRelocSink {
    /// Add a relocation referencing an EBB at the current offset.
    fn reloc_ebb(&mut self, co: CodeOffset, reloc: Reloc, co2: CodeOffset) { unimplemented!() }

    /// Add a relocation referencing an external symbol at the current offset.
    fn reloc_external(&mut self, co: CodeOffset, reloc: Reloc, en: &ExternalName, ad: Addend) { unimplemented!() }

    /// Add a relocation referencing a jump table.
    fn reloc_jt(&mut self, co: CodeOffset, reloc: Reloc, jt: JumpTable) { unimplemented!() }
}
// An opaque reference to variable.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Variable(u32);
impl EntityRef for Variable {
    fn new(index: usize) -> Self {
        assert!(index < (u32::MAX as usize));
        Variable(index as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

#[cfg(unix)]
mod jitter {
    use libc;

    use std::{mem, ptr};

    pub struct Jitter {
        size: usize,
        mem: *mut u8
    }

    impl Jitter {
        pub fn new(bytes: &[u8]) -> Jitter {
            unsafe {
                const PAGE_SIZE: usize = 4096;
                let size = {
                    let mut size = 0;
                    while size < bytes.len() {
                        size += PAGE_SIZE;
                    }
                    size
                };

                // TODO: OS might not give writable + executable memory. Best to ask for writable, then make executable afterwards.
                let mem: *mut u8 = mem::transmute(libc::mmap(
                    ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_ANON | libc::MAP_SHARED,
                    -1,
                    0));

                for (i, x) in bytes.iter().enumerate() {
                    *mem.offset(i as isize) = *x;
                }

                Jitter {
                    size: size,
                    mem: mem
                }
            }
        }

        pub fn run(&mut self) -> i32 {
            unsafe {
                let fn_ptr: extern fn() -> i32 = mem::transmute(self.mem);
                fn_ptr()
            }
        }
        pub fn run_f64_f64_f64(&mut self, a: f64, b: f64) -> f64 {
            unsafe {
                let fn_ptr: extern fn(f64, f64) -> f64 = mem::transmute(self.mem);
                fn_ptr(a, b)
            }
        }
    }

    impl Drop for Jitter {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.mem as *mut _, self.size); }
        }
    }
}

#[cfg(windows)]
mod jitter {
    use winapi;
    use kernel32;

    use std::{mem, ptr};

    pub struct Jitter {
        mem: *mut u8
    }

    impl Jitter {
        pub fn new(bytes: &[u8]) -> Jitter {
            unsafe {
                const PAGE_SIZE: usize = 4096;
                let size = {
                    let mut size = 0;
                    while size < bytes.len() {
                        size += PAGE_SIZE;
                    }
                    size
                };

                // TODO: OS might not give writable + executable memory. Best to ask for writable, then make executable afterwards.
                let mem: *mut u8 = mem::transmute(kernel32::VirtualAlloc(
                    ptr::null_mut(),
                    size as u32,
                    winapi::MEM_COMMIT,
                    winapi::PAGE_EXECUTE_READWRITE));

                for (i, x) in bytes.iter().enumerate() {
                    *mem.offset(i as isize) = *x;
                }

                Jitter {
                    mem: mem
                }
            }
        }

        pub fn run(&mut self) -> i32 {
            unsafe {
                let fn_ptr: extern fn() -> i32 = mem::transmute(self.mem);
                fn_ptr()
            }
        }
    }

    impl Drop for Jitter {
        fn drop(&mut self) {
            unsafe { kernel32::VirtualFree(self.mem as *mut _, 0, winapi::MEM_RELEASE); }
        }
    }
}
