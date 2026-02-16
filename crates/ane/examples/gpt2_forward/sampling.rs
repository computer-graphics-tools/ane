use rand::{Rng, RngExt};

pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    generated_token_ids: &[u32],
    rng: &mut impl Rng,
) -> u32 {
    let mut penalized_logits: Box<[f32]> = logits.into();
    if repetition_penalty != 1.0 {
        for &token_id in generated_token_ids {
            let index = token_id as usize;
            if index < penalized_logits.len() {
                if penalized_logits[index] > 0.0 {
                    penalized_logits[index] /= repetition_penalty;
                } else {
                    penalized_logits[index] *= repetition_penalty;
                }
            }
        }
    }

    if temperature <= 0.0 {
        return penalized_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index as u32)
            .unwrap();
    }

    let max_logit = penalized_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<(usize, f32)> = penalized_logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index, ((logit - max_logit) / temperature).exp()))
        .collect();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let total_probability: f32 = probabilities.iter().map(|(_, prob)| prob).sum();
    let cumulative_cutoff = top_p * total_probability;
    let mut cumulative_sum = 0.0f32;
    let mut candidates = Vec::new();
    for &(token_index, probability) in &probabilities {
        cumulative_sum += probability;
        candidates.push((token_index, probability));
        if cumulative_sum >= cumulative_cutoff {
            break;
        }
    }

    let candidate_total: f32 = candidates.iter().map(|(_, prob)| prob).sum();
    let threshold = rng.random::<f32>() * candidate_total;
    let mut accumulated = 0.0f32;
    for &(token_index, probability) in &candidates {
        accumulated += probability;
        if accumulated >= threshold {
            return token_index as u32;
        }
    }
    candidates.last().map(|(index, _)| *index as u32).unwrap()
}
