// Reduced from the curve25519-dalek 5.0.0 serial backend. See LICENSE.

use std::hint::black_box;
use std::time::Instant;

const MASK51: u64 = (1 << 51) - 1;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Fe([u64; 5]);

#[derive(Copy, Clone)]
#[repr(C)]
pub struct EdwardsPoint {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct AffineNielsPoint {
    y_plus_x: Fe,
    y_minus_x: Fe,
    xy2d: Fe,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct CompletedPoint {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

#[inline(always)]
fn add(a: &Fe, b: &Fe) -> Fe {
    Fe([
        a.0[0] + b.0[0],
        a.0[1] + b.0[1],
        a.0[2] + b.0[2],
        a.0[3] + b.0[3],
        a.0[4] + b.0[4],
    ])
}

#[inline(always)]
fn reduce(mut x: [u64; 5]) -> Fe {
    let c = [x[0] >> 51, x[1] >> 51, x[2] >> 51, x[3] >> 51, x[4] >> 51];
    for limb in &mut x {
        *limb &= MASK51;
    }
    x[0] += c[4] * 19;
    x[1] += c[0];
    x[2] += c[1];
    x[3] += c[2];
    x[4] += c[3];
    Fe(x)
}

#[inline(always)]
fn sub(a: &Fe, b: &Fe) -> Fe {
    reduce([
        a.0[0] + 36_028_797_018_963_664 - b.0[0],
        a.0[1] + 36_028_797_018_963_952 - b.0[1],
        a.0[2] + 36_028_797_018_963_952 - b.0[2],
        a.0[3] + 36_028_797_018_963_952 - b.0[3],
        a.0[4] + 36_028_797_018_963_952 - b.0[4],
    ])
}

#[inline(always)]
fn mul(a: &Fe, b: &Fe) -> Fe {
    #[inline(always)]
    fn m(x: u64, y: u64) -> u128 {
        (x as u128) * (y as u128)
    }

    let b1_19 = b.0[1] * 19;
    let b2_19 = b.0[2] * 19;
    let b3_19 = b.0[3] * 19;
    let b4_19 = b.0[4] * 19;
    let c0 = m(a.0[0], b.0[0])
        + m(a.0[4], b1_19)
        + m(a.0[3], b2_19)
        + m(a.0[2], b3_19)
        + m(a.0[1], b4_19);
    let mut c1 = m(a.0[1], b.0[0])
        + m(a.0[0], b.0[1])
        + m(a.0[4], b2_19)
        + m(a.0[3], b3_19)
        + m(a.0[2], b4_19);
    let mut c2 = m(a.0[2], b.0[0])
        + m(a.0[1], b.0[1])
        + m(a.0[0], b.0[2])
        + m(a.0[4], b3_19)
        + m(a.0[3], b4_19);
    let mut c3 = m(a.0[3], b.0[0])
        + m(a.0[2], b.0[1])
        + m(a.0[1], b.0[2])
        + m(a.0[0], b.0[3])
        + m(a.0[4], b4_19);
    let mut c4 = m(a.0[4], b.0[0])
        + m(a.0[3], b.0[1])
        + m(a.0[2], b.0[2])
        + m(a.0[1], b.0[3])
        + m(a.0[0], b.0[4]);

    let mut out = [0; 5];
    c1 += ((c0 >> 51) as u64) as u128;
    out[0] = (c0 as u64) & MASK51;
    c2 += ((c1 >> 51) as u64) as u128;
    out[1] = (c1 as u64) & MASK51;
    c3 += ((c2 >> 51) as u64) as u128;
    out[2] = (c2 as u64) & MASK51;
    c4 += ((c3 >> 51) as u64) as u128;
    out[3] = (c3 as u64) & MASK51;
    let carry = (c4 >> 51) as u64;
    out[4] = (c4 as u64) & MASK51;
    out[0] += carry * 19;
    out[1] += out[0] >> 51;
    out[0] &= MASK51;
    Fe(out)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn point_add(p: &EdwardsPoint, q: &AffineNielsPoint) -> CompletedPoint {
    let y_plus_x = add(&p.y, &p.x);
    let y_minus_x = sub(&p.y, &p.x);
    let pp = mul(&y_plus_x, &q.y_plus_x);
    let mm = mul(&y_minus_x, &q.y_minus_x);
    let txy2d = mul(&p.t, &q.xy2d);
    let z2 = add(&p.z, &p.z);
    CompletedPoint {
        x: sub(&pp, &mm),
        y: add(&pp, &mm),
        z: add(&z2, &txy2d),
        t: sub(&z2, &txy2d),
    }
}

#[inline(always)]
fn as_extended(p: &CompletedPoint) -> EdwardsPoint {
    EdwardsPoint {
        x: mul(&p.x, &p.t),
        y: mul(&p.y, &p.z),
        z: mul(&p.z, &p.t),
        t: mul(&p.x, &p.y),
    }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn step(p: &mut EdwardsPoint, q: &AffineNielsPoint) {
    *p = as_extended(&point_add(p, q));
}

fn fe(seed: u64) -> Fe {
    Fe(std::array::from_fn(|i| {
        (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15 ^ i as u64)) & MASK51
    }))
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(20_000_000u64);
    let mut p = EdwardsPoint {
        x: fe(1),
        y: fe(2),
        z: fe(3),
        t: fe(4),
    };
    let q = AffineNielsPoint {
        y_plus_x: fe(5),
        y_minus_x: fe(6),
        xy2d: fe(7),
    };
    let start = Instant::now();
    let mut checksum = 0;

    for i in 0..iterations {
        p.x.0[0] = p.x.0[0].wrapping_add(i & 1) & MASK51;
        step(black_box(&mut p), black_box(&q));
        checksum ^= p.x.0[0] ^ p.y.0[1] ^ p.z.0[2] ^ p.t.0[3];
        black_box(&p);
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{:.3} M steps/s, checksum={checksum}",
        iterations as f64 / elapsed / 1e6
    );
}
