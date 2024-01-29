use std::collections::HashSet;

pub fn make_vocabulary(corpus: Vec<&str>) -> Vec<&str> {
    let mut vocabulary: HashSet<&str> = HashSet::new();

    for sentence in corpus {
        let words: Vec<&str> = sentence.split_whitespace().collect();

        for word in words {
            vocabulary.insert(word);
        }
    }
    vocabulary.into_iter().collect::<Vec<&str>>()
}

pub fn make_mock_vocabulary() -> Vec<&'static str> {
    let vocabulary: Vec<&str> = vec![
        "this",
        "is",
        "an",
        "example",
        "vocabulary",
        "just",
        "for",
        "show",
    ];

    vocabulary
}
