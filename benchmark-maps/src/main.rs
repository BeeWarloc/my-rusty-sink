extern crate rand;
extern crate time;
extern crate fnv;

use rand::{Rng,SeedableRng};
use std::collections::HashMap;
use std::collections::BTreeMap;
use fnv::FnvHashMap;

fn benchmark_fnv_hash_map() {
    println!("Benchmarking FnvHashMap");
    let mut map = FnvHashMap::default();
    let mut rng = rand::XorShiftRng::from_seed([5434234, 453426432, 2345234452, 234523453]);
    let start = time::precise_time_s();
    for _ in 0..20_000_000 {
        map.insert(rng.next_u64(), rng.next_u64());
    }
    let total_time = time::precise_time_s() - start;
    println!("{}s elapsed, {}usec per entry. {} entries in map", total_time, (total_time / map.len() as f64) * 1_000_000.0, map.len());
}

fn benchmark_std_hash_map() {
    println!("Benchmarking HashMap");
    let mut map = HashMap::new();
    let mut rng = rand::XorShiftRng::from_seed([5434234, 453426432, 2345234452, 234523453]);
    let start = time::precise_time_s();
    for _ in 0..20_000_000 {
        map.insert(rng.next_u64(), rng.next_u64());
    }
    let total_time = time::precise_time_s() - start;
    println!("{}s elapsed, {}usec per entry. {} entries in map", total_time, (total_time / map.len() as f64) * 1_000_000.0, map.len());
}

fn main() {
    benchmark_fnv_hash_map();
    benchmark_std_hash_map();
    //benchmark_std_btreemap();
}
