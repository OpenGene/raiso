pub struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed as u32;
        for i in 1..N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, mti: N }
    }

    pub fn genrand_uint32(&mut self) -> u32 {
        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            let mut kk = 0;
            while kk < N - M {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
                kk += 1;
            }
            while kk < N - 1 {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk - (N - M)] ^ (y >> 1) ^ mag01[(y & 1) as usize];
                kk += 1;
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn genrand_uint64(&mut self) -> u64 {
        let a = self.genrand_uint32() as u64;
        let b = self.genrand_uint32() as u64;
        (a << 32) | b
    }

    pub fn rk_double(&mut self) -> f32 {
        let a = (self.genrand_uint32() >> 5) as f32;
        let b = (self.genrand_uint32() >> 6) as f32;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    pub fn bounded_below(&mut self, off: u64, rng: u64) -> u64 {
        if rng == 0 {
            return off;
        }

        if rng & (rng - 1) == 0 {
            let mask = rng - 1;
            if mask <= 0xFFFF_FFFF {
                return off + (mask & self.genrand_uint32() as u64);
            }
            return off + (mask & self.genrand_uint64());
        }

        let mask = u64::MAX >> rng.leading_zeros();
        if mask <= 0xFFFF_FFFF {
            loop {
                let val = self.genrand_uint32() as u64 & mask;
                if val < rng {
                    return off + val;
                }
            }
        }
        loop {
            let val = self.genrand_uint64() & mask;
            if val < rng {
                return off + val;
            }
        }
    }

    pub fn shuffle(&mut self, arr: &mut [usize]) {
        let n = arr.len();
        for i in (1..n).rev() {
            let j = self.bounded_below(0, (i + 1) as u64) as usize;
            arr.swap(i, j);
        }
    }
}

#[derive(Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub(crate) fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

pub struct StdNormal {
    rng: SplitMix64,
}

impl StdNormal {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    #[inline]
    fn next_open01(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
    }

    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_open01().max(f32::MIN_POSITIVE);
        let u2 = self.next_open01();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
