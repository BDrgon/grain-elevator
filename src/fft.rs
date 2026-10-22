//! A 1024-point complex FFT, float64, iterative radix-2 (Cooley-Tukey) with cached twiddles.
//!
//! Why hand-rolled: the reference computes `np.fft.rfft(frame * window, 1024)`, and this crate
//! must reproduce it to `1e-4` max-abs through a log-mel and a 512-d embedding. A dependent crate
//! would be fine too, but a fixed-size FFT with a bit-reversal table is ~90 lines, has no
//! features, and can be tested against a fixture once and forgotten.
//!
//! Only `Fft::new(1024)` is used by the frontend (`n_fft` comes from the pinned model spec, and the
//! pinned model is 1024). Other powers of two work.

/// Precomputed plan for one power-of-two size.
pub struct Fft {
    n: usize,
    /// Bit-reversal permutation, `n` entries.
    rev: Vec<u32>,
    /// Twiddle factors `e^{-2 pi i k / n}` for k < n/2, split into re/im.
    tw_re: Vec<f64>,
    tw_im: Vec<f64>,
}

impl Fft {
    /// `n` must be a power of two >= 2.
    pub fn new(n: usize) -> Fft {
        assert!(
            n >= 2 && n.is_power_of_two(),
            "fft size must be a power of two, got {n}"
        );
        let bits = n.trailing_zeros() - 1;
        let mut rev = vec![0u32; n];
        for (i, r) in rev.iter_mut().enumerate() {
            *r = (i as u32).reverse_bits() >> (32 - bits - 1);
        }
        let half = n / 2;
        let mut tw_re = vec![0f64; half];
        let mut tw_im = vec![0f64; half];
        for (k, (wr, wi)) in tw_re.iter_mut().zip(tw_im.iter_mut()).enumerate() {
            // e^{-2*pi*i*k/n}
            let theta = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            *wr = theta.cos();
            *wi = theta.sin();
        }
        Fft {
            n,
            rev,
            tw_re,
            tw_im,
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// In-place forward FFT. `re.len() == im.len() == n`.
    ///
    /// The butterfly form below is the textbook decimation-in-time one; `-` on the imaginary part
    /// of the twiddle makes it a *forward* (analysis) transform, matching numpy.
    pub fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        assert_eq!(re.len(), n, "fft input length");
        assert_eq!(im.len(), n, "fft input length");

        // Bit-reversal permutation (skips the identity swaps).
        for i in 0..n {
            let j = self.rev[i as usize] as usize;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        let mut len = 2usize;
        while len <= n {
            let step = n / len;
            let mut i = 0usize;
            while i < n {
                let mut k = 0usize;
                while k < len / 2 {
                    let a = i + k;
                    let b = a + len / 2;
                    let (wr, wi) = (self.tw_re[k * step], self.tw_im[k * step]);
                    let xr = re[b] * wr - im[b] * wi;
                    let xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                    k += 1;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// `|X[k]|^2` for k = 0..=n/2 — the rfft power spectrum of a real signal, which is all the
    /// log-mel frontend ever looks at. `x.len() == n`; a longer/shorter window in `x` is a bug.
    pub fn rfft_power(&self, x: &[f64], out: &mut [f64]) {
        let n = self.n;
        let bins = n / 2 + 1;
        assert_eq!(x.len(), n, "rfft input length");
        assert!(out.len() >= bins, "rfft power output needs {bins} bins");
        let mut re = x.to_vec();
        let mut im = vec![0f64; n];
        self.forward(&mut re, &mut im);
        for (k, o) in out.iter_mut().take(bins).enumerate() {
            *o = re[k] * re[k] + im[k] * im[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure tone at bin k, sampled the way numpy would, must put all its energy in bin k.
    #[test]
    fn tone_lands_in_one_bin() {
        let n = 1024usize;
        let fft = Fft::new(n);
        for k in [0usize, 1, 7, 64, 256, 511, 512] {
            let x: Vec<f64> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * (k as f64) * (i as f64) / (n as f64)).cos())
                .collect();
            let mut power = vec![0f64; n / 2 + 1];
            fft.rfft_power(&x, &mut power);
            let (best, _) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            assert_eq!(best, k, "peak bin for tone {k}");
            // Parseval: sum |x|^2 == (1/n) sum |X|^2, within float64 slack.
            let time: f64 = x.iter().map(|v| v * v).sum();
            let spec: f64 = power
                .iter()
                .enumerate()
                .map(|(i, p)| if i == 0 || i == n / 2 { *p } else { 2.0 * p })
                .sum();
            assert!(
                (time - spec / n as f64).abs() / time < 1e-9,
                "parseval at bin {k}"
            );
        }
    }

    #[test]
    fn dc_and_impulse_are_known_exactly() {
        let n = 1024usize;
        let fft = Fft::new(n);
        let mut power = vec![0f64; n / 2 + 1];
        fft.rfft_power(&vec![1.0f64; n], &mut power);
        assert_eq!(power[0], (n as f64) * (n as f64));
        for p in &power[1..] {
            assert!(p.abs() < 1e-9, "DC must have no other bins, got {p}");
        }
        let mut x = vec![0f64; n];
        x[3] = 1.0;
        fft.rfft_power(&x, &mut power);
        // |X[k]| = 1 for every bin; power accumulates unnormalized so |X|^2 = 1 too.
        for p in &power {
            assert!((p - 1.0).abs() < 1e-9, "impulse must be flat, got {p}");
        }
    }
}
