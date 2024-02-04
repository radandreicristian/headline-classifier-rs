#[cfg(test)]
mod test_metrics {

    use candle_core::{Device, Tensor};
    use common::metrics::*;

    #[test]
    fn test_true_positives() {
        let predicted = vec![vec![1., 0.], vec![0., 1.]];
        let actual = vec![vec![1., 0.], vec![1., 0.]];

        let actual_result = true_positives(&predicted, &actual);
        let expected_result = 1;

        assert_eq!(expected_result, actual_result);
    }

    #[test]
    fn test_false_positives() {
        let predicted = vec![vec![1., 0.], vec![0., 1.]];
        let actual = vec![vec![1., 0.], vec![1., 0.]];

        let actual_result = false_positives(&predicted, &actual);
        let expected_result = 1;

        assert_eq!(expected_result, actual_result);
    }

    #[test]
    fn test_false_negatives() {
        let predicted = vec![vec![1., 0.], vec![0., 1.]];
        let actual = vec![vec![1., 0.], vec![1., 0.]];

        let actual_result = false_negatives(&predicted, &actual);
        let expected_result = 1;

        assert_eq!(expected_result, actual_result);
    }
}
