#[cfg(test)]
mod encode {

    use common::vocabulary::*;

    #[test]
    fn test_make_vocabulary_single_sentence() {
        let corpus = vec!["Hello, world!".to_string()];
        let vocabulary = make_vocabulary(&corpus);

        let expected_words_in_vocabulary = vec!["Hello".to_string(), "world".to_string()];
        println!("{:?}", vocabulary);
        for word in expected_words_in_vocabulary {
            assert!(vocabulary.contains(&word));
        }
    }

    #[test]
    fn test_make_vocabulary_multiple_sentences() {
        let corpus = vec![
            "This is a test.".to_string(),
            "Another test.".to_string(),
        ];
        let vocabulary = make_vocabulary(&corpus);
        let expected_words_in_vocabulary: Vec<String> = vec!["This".to_string(), "is".to_string(), "a".to_string(), "test".to_string(), "Another".to_string()];
        for word in expected_words_in_vocabulary {
            assert!(vocabulary.contains(&word));
        }
    }
}