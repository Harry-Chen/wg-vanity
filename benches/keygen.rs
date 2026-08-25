use std::hint::black_box;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use x25519_dalek::{PublicKey, StaticSecret};

use wg_vanity::{PatternKind, SearchPattern, trial};

fn candidate_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("candidate");
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    let public_b64 = STANDARD.encode(public.as_bytes());

    group.bench_function("private_key", |b| b.iter(StaticSecret::random));
    group.bench_function("x25519_public_key", |b| {
        b.iter(|| PublicKey::from(black_box(&private)))
    });
    group.bench_function("base64_public_key", |b| {
        b.iter(|| STANDARD.encode(black_box(public.as_bytes())))
    });
    group.bench_function("case_insensitive_match", |b| {
        b.iter(|| {
            black_box(&public_b64[0..10])
                .to_ascii_lowercase()
                .contains("****")
        })
    });
    let literal = SearchPattern::new("zzzzzzzzzz", PatternKind::Literal, false).unwrap();
    let regex = SearchPattern::new("zzzzzzzzzz", PatternKind::Regex, false).unwrap();
    let glob = SearchPattern::new("zz*zzzzzz", PatternKind::Glob, false).unwrap();
    let glob_regex = SearchPattern::new("zz.*zzzzzz", PatternKind::Regex, false).unwrap();
    group.bench_function("literal_match_same_pattern", |b| {
        b.iter(|| literal.is_match(black_box(public_b64.as_bytes()), 0, 10))
    });
    group.bench_function("regex_match_same_pattern", |b| {
        b.iter(|| regex.is_match(black_box(public_b64.as_bytes()), 0, 10))
    });
    group.bench_function("glob_match_same_pattern", |b| {
        b.iter(|| glob.is_match(black_box(public_b64.as_bytes()), 0, 10))
    });
    group.bench_function("glob_regex_match_same_pattern", |b| {
        b.iter(|| glob_regex.is_match(black_box(public_b64.as_bytes()), 0, 10))
    });
    group.bench_function("complete_no_match", |b| {
        b.iter(|| trial(black_box("****"), 0, 10))
    });
    group.finish();
}

fn cpu_search_batches(c: &mut Criterion) {
    const BATCH_SIZES: [u64; 3] = [1, 1_000, 100_000];
    let mut group = c.benchmark_group("cpu_search");
    for batch in BATCH_SIZES {
        group.throughput(Throughput::Elements(batch));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch| {
            b.iter(|| {
                (0..batch)
                    .into_par_iter()
                    .filter_map(|_| trial("****", 0, 10))
                    .count()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, candidate_stages, cpu_search_batches);
criterion_main!(benches);
