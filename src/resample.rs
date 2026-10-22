//! `scipy.signal.resample_poly` in Rust, matching the reference's resampler (engine.py:40:
//! `g=gcd(sr,48000); signal=resample_poly(signal,48000//g,sr//g)`).
//!
//! Not a "good enough" resampler: the exact algorithm, including the filter scipy designs
//! (`firwin(2*10*max(up,down)+1, 1/max(up,down), window=('kaiser',5.0))`, then `h *= up`), the
//! pre-padding `down - half_len % down`, the post-padding loop, and the sample count
//! `n_pre_remove` it throws away. Those pad/removal counts are what put a sample at the right
//! index; an off-by-one here shifts every mel frame and the §9 gate fails no matter how good the
//! filter is.
//!
//! Tested directly against `tools/fixtures/out/resample/` (11 rates including 11025 → 640/147 and
//! 12345 → 3200/823, the two longest filters), not just end to end.

/// Kaiser-windowed lowpass FIR, `numtaps` odd, cutoff expressed relative to Nyquist (scipy's
/// `fs=2` convention): `firwin(numtaps, f_c, window=('kaiser', beta))`.
pub fn firwin_kaiser(numtaps: usize, f_c: f64, beta: f64) -> Vec<f64> {
    assert!(numtaps >= 3, "filter needs at least 3 taps");
    let alpha = (numtaps - 1) as f64 / 2.0;
    let mut h = vec![0.0f64; numtaps];
    for (n, v) in h.iter_mut().enumerate() {
        let m = (n as f64) - alpha;
        // firwin: h = 2*cutoff/fs * sinc(2*cutoff/fs*m), with fs = 2 → f_c * sinc(f_c*m).
        let x = f_c * m;
        let s = sinc(x);
        *v = f_c * s * kaiser_window_value(n, numtaps, beta);
    }
    // Type I (odd length, even symmetry): normalise to unity DC gain, as firwin does.
    let sum: f64 = h.iter().sum();
    if sum != 0.0 {
        for v in h.iter_mut() {
            *v /= sum;
        }
    }
    h
}

/// `numpy.sinc`: sin(pi x) / (pi x), with the removable singularity at 0.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Symmetric Kaiser window sample: `I0(beta*sqrt(1-t^2)) / I0(beta)` with `t = (n-alpha)/alpha`.
fn kaiser_window_value(n: usize, numtaps: usize, beta: f64) -> f64 {
    let alpha = (numtaps - 1) as f64 / 2.0;
    let t = ((n as f64) - alpha) / alpha;
    let inside = 1.0 - t * t;
    if inside <= 0.0 {
        // The endpoints: sqrt(0) -> I0(0)/I0(beta).
        return bessel_i0(0.0) / bessel_i0(beta);
    }
    bessel_i0(beta * inside.sqrt()) / bessel_i0(beta)
}

/// Modified Bessel function of the first kind, order 0, in float64.
///
/// `I0(x) = sum_{k>=0} ((x/2)^(2k)) / (k!)^2`. The series converges in a few dozen terms for the
/// arguments used here (`beta = 5`), and stays within a couple of ULPs — the usual polynomial
/// approximations are ~1e-7 and that error would land directly in the filter coefficients.
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let q = (ax / 2.0) * (ax / 2.0);
    for k in 1..200usize {
        let kf = k as f64;
        term *= q / (kf * kf);
        sum += term;
        if term < sum * 1e-18 {
            break;
        }
    }
    sum
}

/// `upfirdn` output length, as measured against scipy (see `resample/meta.json`): the
/// `ceil((in*up + h_len - 1)/down) + 1` in the docstring is not what the code returns.
fn upfirdn_out_len(h_len: usize, in_len: usize, up: usize, down: usize) -> usize {
    if in_len == 0 {
        return 0;
    }
    ((in_len - 1) * up + h_len - 1) / down + 1
}

/// Rational resampling by `up/down`, zero-padded edges (`mode='constant', cval=0`).
///
/// The convolution runs per output sample over the taps that actually land on a real input sample
/// (a polyphase walk), so the cost is `out_frames * taps/upto` rather than `out_frames * taps`:
/// 48 k out at 11025→48000 is ~12801 taps naively, ~20 per output sample this way.
pub fn resample_poly(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    let (up, down) = reduce(up, down);
    if up == 1 && down == 1 || x.is_empty() {
        return x.to_vec();
    }
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    // scipy designs the filter in float64, casts it to the input dtype (float32 here), then
    // multiplies by `up`. Doing the same keeps the *coefficients* bit-identical to the reference;
    // only the accumulation is wider than scipy's, which is strictly an improvement.
    let mut taps32: Vec<f32> = firwin_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, 5.0)
        .into_iter()
        .map(|v| (v as f32) * (up as f32))
        .collect();

    let n_in = x.len();
    let n_out = (n_in * up + down - 1) / down; // ceil(in*up/down)

    let n_pre_pad = down - half_len % down;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    let mut n_post_pad = 0usize;
    let h_len0 = taps32.len();
    while upfirdn_out_len(h_len0 + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut taps = Vec::with_capacity(n_pre_pad + h_len0 + n_post_pad);
    taps.extend(std::iter::repeat_n(0.0f32, n_pre_pad));
    taps.append(&mut taps32);
    taps.extend(std::iter::repeat_n(0.0f32, n_post_pad));
    let h_len = taps.len();

    // Conceptual input position for output index `n`, in upsampled time: the filter's first tap
    // lines up with input sample index `-((h_len - 1))` shifted by the padding scipy added, so the
    // walk below starts from the first input sample a tap actually touches.
    let mut out = Vec::with_capacity(n_out);
    for o in 0..n_out {
        let t = ((o + n_pre_remove) * down) as i64; // position in upsampled/filtered time
        let mut acc = 0.0f64;
        // Tap k samples input j where (t - k) is a multiple of up: j = (t - k)/up.
        // k ranges so that 0 <= j < n_in.
        let k_first = {
            let k = t.rem_euclid(up as i64);
            k
        };
        let mut k = k_first;
        while k < h_len as i64 {
            let j = (t - k) / up as i64;
            if j >= 0 && (j as usize) < n_in {
                acc += taps[k as usize] as f64 * x[j as usize] as f64;
            }
            k += up as i64;
        }
        out.push(acc as f32);
    }
    out
}

/// `resample_poly` with the ratio already reduced; also used by the tests.
fn reduce(up: usize, down: usize) -> (usize, usize) {
    let g = gcd(up, down).max(1);
    (up / g, down / g)
}

pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Resample `x` from `src_rate` to `dst_rate` the way the reference does.
pub fn resample_to(x: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return x.to_vec();
    }
    resample_poly(x, dst_rate as usize, src_rate as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_i0_matches_known_values() {
        // Reference values, float64.
        for (x, want) in [
            (0.0f64, 1.0),
            (1.0, 1.2660658777520082),
            (5.0, 27.239871823604442),
        ] {
            let got = bessel_i0(x);
            assert!(
                (got - want).abs() / want < 1e-12,
                "i0({x}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn identity_and_simple_up_down() {
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(resample_poly(&x, 1, 1), x);
        // Exact by construction: downsampling by 1 after upsampling by 1 keeps the length.
        let y = resample_poly(&x, 2, 1);
        assert_eq!(y.len(), 32);
        let z = resample_poly(&x, 1, 2);
        assert_eq!(z.len(), 8);
    }

    #[test]
    fn gcd_reduces_before_design() {
        // 96000 -> 48000 must reduce to 1/2, not design a 96000-tap filter.
        assert_eq!(reduce(48_000, 96_000), (1, 2));
        assert_eq!(reduce(48_000, 11_025), (640, 147));
        assert_eq!(reduce(48_000, 44_100), (160, 147));
        assert_eq!(reduce(48_000, 12_345), (3200, 823));
    }

    #[test]
    fn upfirdn_length_matches_the_measured_formula() {
        // From tools/fixtures/out/resample/meta.json, measured against scipy.
        assert_eq!(upfirdn_out_len(7, 10, 2, 3), 9);
        assert_eq!(upfirdn_out_len(3, 1, 1, 1), 3);
        assert_eq!(upfirdn_out_len(5, 4, 3, 2), 7);
        assert_eq!(upfirdn_out_len(3201, 16538, 160, 147), 18022);
    }
}
