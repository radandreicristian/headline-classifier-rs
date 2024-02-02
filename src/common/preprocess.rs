/// Pads a vector with a specified padding value to reach a maximum length.
///
/// This function takes an input vector, a maximum padding length, and a padding value. It then
/// modifies the input vector to ensure that it has the maximum length, either by truncating it
/// if it exceeds the maximum length or by adding elements with the padding value if it falls
/// short of the maximum length.
///
/// # Arguments
///
/// * `vector`: The input vector that you want to pad.
/// * `max_padding`: The maximum length that the input vector should have after padding.
/// * `pad_value`: The value to use for padding when extending the vector.
///
/// # Returns
///
/// A new vector that has been padded to reach the specified maximum length.
///
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
