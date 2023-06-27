fn main() {
    // What about the issue that arises when we calculate an index of a mutable,
    // what happens if the mutable changes??
    //
    let mut s = String::from("Yo");
    let word = first_word(&s);
    println!("BEFORE EMPTY: s: {}, word: {}", s, word);
    s.clear();  // Empty the string
    println!("AFTER EMPTY: s: {}, word: {}", s, word);

    // Variables getting out of sync is a common bug... how can Slices fix this?

    let mut s = String::from("foo bar");
    let foo = &s[0..3];  // Reference to a **portion** of the string
    let bar = &s[4..7];  // Incl. start, excl. end (no start = 0, no end = len)
    println!("s: '{}', foo: '{}', bar: '{}'", s, foo, bar);
    s.clear();  // Empty the string
    // ERROR: After emptying s, compile error occurs when attempting to use slice
    // println!(" AFTER CLEAR: s: '{}', foo: '{}', bar: '{}'", s, foo, bar);

    // Now lest re-write first word to work properly with slices
    let s = String::from("foo bar baz");
    let word = first_word_w_slices(&s);
    println!("s: '{}', word: '{}'", s, word);

    // Probably makes more sense to write an API that outpus &str with input &str
    let word = first_word_w_ref_input(&s);
    println!("s: '{}', word: '{}'", s, word);
    let word = first_word_w_ref_input(&s[..]);
    println!("s: '{}', word: '{}'", s, word);
    let word = first_word_w_ref_input(&s[..2]);
    println!("s: '{}', word: '{}'", s, word);

    // Slices also work with arrays in the obvious way
}

fn first_word(s: &String) -> usize {
    // Read through the values byte by byte
    // and get the index of the first space
    let bytes = s.as_bytes();

    // Create an iterator with the `.iter()` method
    // the item returned by the iterator is also a refernce, so & is needed
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return i;  // Use the `return` kword to do an early return
        }
    }

    // Otherwise return the length of the string
    s.len()
}

fn first_word_w_slices(s: &String) -> &str {
    // Whats up with the return type??
    // Refernce to a str literal
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[..i];
        }
    }
    &s[..]
}

// This version works with bother `String` refs and slices of strings
// (Note that &String == &s[..], which are string literals!!)
// see "deref corecions", (ch15)
fn first_word_w_ref_input(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[..i];
        }
    }
    &s[..]

}
