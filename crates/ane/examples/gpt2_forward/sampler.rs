use rand::{Rng, RngExt};

pub struct Sampler {
    logits: Vec<f32>,
    probabilities: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            logits: Vec::new(),
            probabilities: Vec::new(),
        }
    }

    pub fn sample(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        history: &[u32],
        rng: &mut impl Rng,
    ) -> u32 {
        self.logits.clear();
        self.logits.extend_from_slice(logits);
        for &token in history {
            if repetition_penalty != 1.0
                && let Some(value) = self.logits.get_mut(token as usize)
            {
                if *value > 0.0 {
                    *value /= repetition_penalty;
                } else {
                    *value *= repetition_penalty;
                }
            }
        }
        if temperature <= 0.0 {
            return self
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("non-finite model scores"))
                .unwrap()
                .0 as u32;
        }
        let max = self
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        self.probabilities.clear();
        self.probabilities
            .extend(self.logits.iter().enumerate().map(|(i, &x)| {
                let probability = ((x - max) / temperature).exp();
                assert!(probability.is_finite(), "non-finite model scores");
                (i as u32, probability)
            }));
        let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        let cutoff = top_p as f64 * self.probabilities.iter().map(|p| p.1 as f64).sum::<f64>();
        let mut count = self.probabilities.len().min(256);
        let mut fixed = 0;
        let mut mass = 0.0;
        loop {
            if count < self.probabilities.len() {
                self.probabilities[fixed..].select_nth_unstable_by(count - fixed - 1, order);
            }
            mass += self.probabilities[fixed..count]
                .iter()
                .map(|p| p.1 as f64)
                .sum::<f64>();
            if mass >= cutoff || count == self.probabilities.len() {
                break;
            }
            fixed = count;
            count = (count * 2).min(self.probabilities.len());
        }
        self.probabilities[..count].sort_unstable_by(order);
        mass = 0.0;
        let count = self.probabilities[..count]
            .iter()
            .position(|p| {
                mass += p.1 as f64;
                mass >= cutoff
            })
            .map_or(count, |i| i + 1);
        let threshold = rng.random::<f32>() as f64 * mass;
        let mut cumulative = 0.0;
        self.probabilities[..count]
            .iter()
            .find(|p| {
                cumulative += p.1 as f64;
                cumulative >= threshold
            })
            .unwrap_or(&self.probabilities[count - 1])
            .0
    }
}
