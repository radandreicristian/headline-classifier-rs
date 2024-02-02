#[cfg(test)]
mod encode {

    use std::collections::HashMap;

    use common::encode::*;

    #[test]
    fn test_create_vocabulary_to_index_mapping_multiple_words() {
        let vocabulary = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let result = create_vocabulary_to_index_mapping(&vocabulary);
        let expected: HashMap<String, u32> = [
            ("<UNK>".to_string(), 0),
            ("apple".to_string(), 1),
            ("banana".to_string(), 2),
            ("cherry".to_string(), 3),
        ]
        .iter()
        .cloned()
        .collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_map_to_indices_word_not_in_mapping() {
        let words = vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()];
        let mapping: HashMap<String, u32> = [("apple".to_string(), 1), ("cherry".to_string(), 3)]
            .iter()
            .cloned()
            .collect();

        let result = map_to_indices(words, &mapping);
        let expected: Vec<u32> = vec![1, 0, 3];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_map_to_indices_empty_words() {
        let words: Vec<String> = Vec::new();
        let mapping: HashMap<String, u32> = HashMap::new();

        let result = map_to_indices(words, &mapping);
        let expected: Vec<u32> = Vec::new();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_map_to_indices_word_in_mapping() {
        let words = vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()];
        let mapping: HashMap<String, u32> = [
            ("apple".to_string(), 1),
            ("banana".to_string(), 2),
            ("cherry".to_string(), 3),
        ]
        .iter()
        .cloned()
        .collect();

        let result = map_to_indices(words, &mapping);
        let expected: Vec<u32> = vec![1, 2, 3];

        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_create_class_mapping_empty_labels() {
        let labels: Vec<String> = Vec::new();

        let (class_to_index, index_to_class) = create_class_mapping_from_labels(&labels);

        let expected_class_to_index: HashMap<String, u32> = HashMap::new();
        let expected_index_to_class: HashMap<u32, String> = HashMap::new();

        assert_eq!(class_to_index, expected_class_to_index);
        assert_eq!(index_to_class, expected_index_to_class);
    }

    #[test]
    fn test_create_class_mapping_single_label() {
        let labels = vec!["ClassA".to_string()];

        let (class_to_index, index_to_class) = create_class_mapping_from_labels(&labels);

        let mut expected_class_to_index: HashMap<String, u32> = HashMap::new();
        let mut expected_index_to_class: HashMap<u32, String> = HashMap::new();

        expected_class_to_index.insert("ClassA".to_string(), 0);
        expected_index_to_class.insert(0, "ClassA".to_string());

        assert_eq!(class_to_index, expected_class_to_index);
        assert_eq!(index_to_class, expected_index_to_class);
    }

    #[test]
    fn test_create_class_mapping_multiple_labels() {
        let labels = vec!["ClassA|ClassB".to_string(), "ClassC".to_string(), "".to_string()];

        let (class_to_index, index_to_class) = create_class_mapping_from_labels(&labels);

        let mut expected_class_to_index: HashMap<String, u32> = HashMap::new();
        let mut expected_index_to_class: HashMap<u32, String> = HashMap::new();

        expected_class_to_index.insert("ClassA".to_string(), 0);
        expected_class_to_index.insert("ClassB".to_string(), 1);
        expected_class_to_index.insert("ClassC".to_string(), 2);

        expected_index_to_class.insert(0, "ClassA".to_string());
        expected_index_to_class.insert(1, "ClassB".to_string());
        expected_index_to_class.insert(2, "ClassC".to_string());

        assert_eq!(class_to_index, expected_class_to_index);
        assert_eq!(index_to_class, expected_index_to_class);
    }
    #[test]
    fn test_multi_hot_encode_empty_labels() {
        let labels: Vec<String> = Vec::new();
        let class_to_index: HashMap<String, u32> = HashMap::new();

        let result = multi_hot_encode(labels, &class_to_index);

        match result {
            Ok(encodings) => assert_eq!(encodings, Vec::<u32>::new()),
            _ => panic!("Expected Ok(Vec::new())"),
        }
    }

    #[test]
    fn test_multi_hot_encode_single_label_not_found() {
        let labels = vec!["ClassA".to_string()];
        let class_to_index: HashMap<String, u32> = HashMap::new();

        let result = multi_hot_encode(labels, &class_to_index);

        match result {
            Err(err) => assert_eq!(
                err.to_string(),
                "Label not found: ClassA".to_string()
            ),
            _ => panic!("Expected Err(\"Label not found: ClassA\")"),
        }
    }

    #[test]
    fn test_multi_hot_encode_single_label_found() {
        let labels = vec!["ClassA".to_string()];
        let mut class_to_index: HashMap<String, u32> = HashMap::new();
        class_to_index.insert("ClassA".to_string(), 0);

        let result = multi_hot_encode(labels, &class_to_index);

        match result {
            Ok(encodings) => assert_eq!(encodings, vec![1]),
            _ => panic!("Expected Ok(vec![1])"),
        }
    }

    #[test]
    fn test_multi_hot_encode_multiple_labels() {
        let labels = vec!["ClassA|ClassB".to_string(), "ClassC".to_string(), "".to_string()];
        let mut class_to_index: HashMap<String, u32> = HashMap::new();
        class_to_index.insert("ClassA".to_string(), 0);
        class_to_index.insert("ClassB".to_string(), 1);
        class_to_index.insert("ClassC".to_string(), 2);

        let result = multi_hot_encode(labels, &class_to_index);

        match result {
            Ok(encodings) => assert_eq!(
                encodings,
                vec![
                    1, 1, 0, // ClassA, ClassB, empty label
                    0, 0, 1, // ClassC, empty label, ClassC
                    0, 0, 0, // empty label, empty label, empty label
                ]
            ),
            _ => panic!("Expected Ok(encodings)"),
        }
    }

}
