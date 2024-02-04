use candle_core::Tensor;

pub fn f1_score(predicted_labels: &Tensor, actual_labels: &Tensor) -> Result<f32, candle_core::Error> {
    let predicted_vector = predicted_labels.to_vec2::<f32>()?;
    let actual_vector = actual_labels.to_vec2::<f32>()?;

    let true_positives = true_positives(&predicted_vector, &actual_vector);

    if true_positives == 0 {
        return Ok(0.);
    }

    let false_positives = false_positives(&predicted_vector, &actual_vector);
    let false_negatives = false_negatives(&predicted_vector, &actual_vector);

    let precision = true_positives as f32 / (true_positives + false_positives) as f32;
    let recall = true_positives as f32 / (true_positives + false_negatives) as f32;

    let f1_score = 2.0 * (precision * recall) / (precision + recall);

    Ok(f1_score)
}


fn fold_with_values<T: PartialEq>(predicted_labels: &Vec<Vec<T>>, actual_labels: &Vec<Vec<T>>, predicted_value: T, actual_value: T) -> usize {
        
    predicted_labels.iter().zip(actual_labels.iter()).fold(0, |acc, (p, a)| {
        acc + p.iter().zip(a.iter()).filter(|(p_i, a_i)| **p_i == predicted_value && **a_i == actual_value).count()
    })
}

pub fn true_positives(predicted_labels: &Vec<Vec<f32>>, actual_labels: &Vec<Vec<f32>>) -> usize {
    fold_with_values(predicted_labels, actual_labels, 1., 1.)
}

pub fn false_positives(predicted_labels: &Vec<Vec<f32>>, actual_labels: &Vec<Vec<f32>>) -> usize {
    fold_with_values(predicted_labels, actual_labels, 1., 0.)
}

pub fn false_negatives(predicted_labels: &Vec<Vec<f32>>, actual_labels: &Vec<Vec<f32>>) -> usize {
    fold_with_values(predicted_labels, actual_labels, 0., 1.)
}