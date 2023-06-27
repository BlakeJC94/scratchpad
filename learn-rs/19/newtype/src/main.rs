// This is a pattern that allows us to bypass the restriction of
// "Only traits implemented on types if either type of trait is crate local"
// (No runtime performance incurred)

// For example let's implement `Display` on `Vec<T>`.
// Neither of these are crate local, but we can create a small wrapper to do this
use std::fmt;

struct Wrapper(Vec<String>);  // Define a tuple type

impl fmt::Display for Wrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}]", self.0.join(", "))
    }
}

fn main() {
    let w = Wrapper(vec![String::from("foo"), String::from("bar")]);
    println!("w = {}", w);
}

// Downside is that this wrapper is technically a new type, after all
