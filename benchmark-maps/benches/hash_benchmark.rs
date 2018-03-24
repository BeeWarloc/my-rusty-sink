#[macro_use]
extern crate criterion;

use criterion::Criterion;

fn run_hash_map() {
    
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci());
}
