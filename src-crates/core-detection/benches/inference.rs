#![allow(missing_docs)]
//! Benchmarks for magika file-type detection.
use std::fs;

#[path = "../tests/mod.rs"]
mod tests;

use akuna_core_detection::Session;
use burn_cpu::CpuDevice;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
fn benchmark_inference(c: &mut Criterion) {
    let fixture_paths = tests::fixture_files();
    let fixture_bytes = fixture_paths
        .iter()
        .map(|path| fs::read(path).expect("read fixture"))
        .collect::<Vec<_>>();

    let session =
        Session::<burn_cpu::Cpu>::new(&CpuDevice).expect("build session");

    let first = fixture_bytes.first().expect("at least one fixture").clone();
    c.bench_function("detect_bytes_single", |b| {
        b.iter_batched(
            || first.clone(),
            |bytes| {
                session
                    .detect_content_sync(bytes.as_slice())
                    .expect("single inference")
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("detect_batch_fixtures", |b| {
        b.iter(|| {
            let batch = fixture_bytes
                .iter()
                .map(|bytes| bytes.as_slice())
                .collect::<Vec<_>>();
            session
                .detect_content_batch_sync(batch)
                .expect("batch inference")
        })
    });
}

criterion_group!(benches, benchmark_inference);

criterion_main!(benches);
