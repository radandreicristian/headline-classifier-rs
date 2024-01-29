use std::collections::HashSet;

pub fn make_vocabulary(corpus: &Vec<String>) -> Vec<String> {
    let mut vocabulary: HashSet<String> = HashSet::new();

    for sentence in corpus {
        let words: Vec<&str> = sentence.split_whitespace().collect();

        for word in words {
            vocabulary.insert(word.to_string());
        }
    }
    vocabulary.into_iter().collect::<Vec<String>>()
}

pub fn make_mock_vocabulary() -> Vec<String> {
    let vocabulary: Vec<String> = vec![
        "this",
        "is",
        "an",
        "example",
        "vocabulary",
        "just",
        "for",
        "show",
    ]
    .iter()
    .map(|word| word.to_string())
    .collect();

    vocabulary
}
