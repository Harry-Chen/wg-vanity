use base64::{Engine as _, engine::general_purpose::STANDARD};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use x25519_dalek::{PublicKey, StaticSecret};

use wireguard_vanity_lib::trial;

fn b1_point_generation(c: &mut Criterion) {
    c.bench_function("b1_point_generation", |b| b.iter(StaticSecret::random));
}

fn b2a_point_conversion(c: &mut Criterion) {
    let private = StaticSecret::random();
    c.bench_function("b2a_point_conversion", |b| {
        b.iter(|| PublicKey::from(&private))
    });
}

fn b2b_point_to_bytes(c: &mut Criterion) {
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    c.bench_function("b2b_point_to_bytes", |b| b.iter(|| public.as_bytes()));
}

fn b2c_bytes_to_base64(c: &mut Criterion) {
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    let public_bytes = public.as_bytes();
    c.bench_function("b2c_bytes_to_base64", |b| {
        b.iter(|| STANDARD.encode(black_box(public_bytes)))
    });
}

fn b2d_base64_contains(c: &mut Criterion) {
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    let public_b64 = STANDARD.encode(public.as_bytes());
    c.bench_function("b2d_base64_contains", |b| {
        b.iter(|| public_b64[0..10].to_ascii_lowercase().contains("****"))
    });
}

fn b2e_total_point_checking(c: &mut Criterion) {
    c.bench_function("b2e_total_point_checking", |b| {
        b.iter(|| {
            let private = StaticSecret::random();
            let public = PublicKey::from(&private);
            let public_b64 = STANDARD.encode(public.as_bytes());
            public_b64[0..10].to_ascii_lowercase().contains("****")
        })
    });
}

fn b3_point_generation_and_checking(c: &mut Criterion) {
    let prefix: &str = "****";
    c.bench_function("b3_point_generation_and_checking", |b| {
        b.iter(|| trial(&prefix, 0, 10))
    });
}

criterion_group!(
    benches,
    b1_point_generation,
    b2a_point_conversion,
    b2b_point_to_bytes,
    b2c_bytes_to_base64,
    b2d_base64_contains,
    b2e_total_point_checking,
    b3_point_generation_and_checking,
);
criterion_main!(benches);
