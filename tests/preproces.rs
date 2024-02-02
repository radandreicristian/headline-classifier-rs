mod encode;

#[cfg(test)]
mod test_convert {

    use common::preprocess::*;

    #[test]
    fn test_pad_vector_smaller() {
        let vector: Vec<u32> = Vec::new();
        let max_padding = 5_usize;
        let pad_value = 1_u32;

        let expected_result: Vec<u32> = vec![1; 5];
        let actual_result = pad_vector(vector, max_padding, pad_value);

        assert_eq!(expected_result, actual_result);
    }

    #[test]
    fn test_pad_vector_larger() {
        let vector: Vec<u32> = vec![1; 5];
        let max_padding = 4;
        let pad_value = 1;

        let expected_result: Vec<u32> = vec![1; 4];
        let actual_result = pad_vector(vector, max_padding, pad_value);

        assert_eq!(expected_result, actual_result);
    }

    #[test]
    fn test_pad_vector_equal() {
        let vector: Vec<u32> = vec![1; 5];
        let max_padding = 5;
        let pad_value = 1;
        let expected_result: Vec<u32> = vec![1; 5];

        let actual_result = pad_vector(vector, max_padding, pad_value);

        assert_eq!(expected_result, actual_result);
    }
}
