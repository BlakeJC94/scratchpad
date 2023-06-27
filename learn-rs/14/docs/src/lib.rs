//! # Foo
//!
//! `docs` is a collection of utilities to illustrate how documentation
//! works in Rust (with basic functions).

// Docs for the head of the file are indicated with //!


// Documentation is indicated with ///, supports markdown
// and is automatically compiled to html with `$ cargo doc` in `./target/doc`
//
// `$ cargo doc --open` will open build and open the docs

/// Adds one to the number given
///
/// # Examples
///
/// ```
/// let arg = 5;
/// let answer = docs::add_one(arg);
///
/// assert_eq!(6, answer)
/// ```
pub fn add_one(x: usize) -> usize {
    x + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add_one(2);
        assert_eq!(result, 3);
    }
}

// In addition to `# Examples`, good docs should also document
// `# Panics`, `# Errors`, `# Safety`


// We can use `pub use` statements in a crate to make the API more usable
// and create more discoverable docs

// Without these statements, the docs would remain hierarchical
// and the compenents we want users to see are not on the main page
pub use self::kinds::PrimaryColor;
pub use self::kinds::SecondaryColor;
pub use self::utils::mix;


pub mod kinds {
    /// The primary colors according to the RYB color model.
    pub enum PrimaryColor {
        Red,
        Yellow,
        Blue,
    }

    /// The secondary colors according to the RYB color model.
    pub enum SecondaryColor {
        Orange,
        Green,
        Purple,
    }
}

pub mod utils {
    use crate::kinds::*;

    /// Combines two primary colors in equal amounts to create
    /// a secondary color.
    pub fn mix(c1: PrimaryColor, c2: PrimaryColor) -> SecondaryColor {
        // --snip--
        unimplemented!();
    }
}

