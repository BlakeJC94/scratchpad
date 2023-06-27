use std::env::args;

const VOWELS: [char; 5] = ['a', 'e', 'i', 'o', 'u'];

fn main() {
    let pig_sentence = convert_sentence_input();
    println!("{}", pig_sentence);
}

fn convert_sentence_input() -> String {
    let mut pig_sentence = String::new();
    for (i, word) in args().enumerate() {
        if i == 0 {
            continue;
        }

        let pig_word = convert_word(word);
        pig_sentence = pig_sentence + &pig_word;

        if i < args().len() - 1 {
            pig_sentence.push(' ');
        }
    }
    pig_sentence
}

// If word begins with vowel, append "-hay" and return
// Else drop first letter and append -_ay
fn convert_word(word: String) -> String {
    let mut word = word.clone();
    let first_char = word.chars().next();

    let first_letter: char;
    first_letter = match first_char {
        Some(letter) => letter,
        _ => '_',
    };

    let mut suffix = String::new();
    if is_vowel(first_letter) {
        suffix.push('h');
    } else {
        suffix.push(first_letter);
        word.remove(0);
    }
    suffix = suffix + "ay";

    format!("{}-{}", word, suffix)
}

fn is_vowel(l: char) -> bool {
    let mut out = false;
    for v in &VOWELS {
        if l == *v {
            out = true;
            break;
        }
    }
    out
}
