use super::exception::InferenceError;

pub fn pad_vector<T: Clone>(mut vector: Vec<T>, max_padding: usize, pad_value: T) -> Vec<T> {
    match vector.len() {
        len if len > max_padding => vector.truncate(max_padding),
        len if len < max_padding => {
            vector.extend(std::iter::repeat(pad_value).take(max_padding - len))
        }
        _ => (),
    }
    vector
}

pub fn convert_to_array<T, const N: usize>(vec: Vec<T>) -> Result<[T; N], InferenceError>
where
    T: Default + Clone,
{
    vec.try_into()
        .map_err(|_| InferenceError::ArrayConversionError("Could not convert to array."))
}
