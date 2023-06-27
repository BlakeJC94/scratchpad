// Define a module with the `mod` kword
// `mod` will look for code in braces, or a file
mod front_of_house {
    pub mod hosting {  // FIX: append `pub` to access module tree branches and leaves
        pub fn add_to_waitlist() {}
        pub fn seat_at_table() {}
    }
    pub mod serving {
        pub fn take_order() {}
        pub fn serve_order() {}
        fn take_payment() {}
    }
}

// `src/lib.rs` is the crate root, with the following module tree
// crate   <-- Implicit module
//  └── front_of_house
//      ├── hosting
//      │   ├── add_to_waitlist
//      │   └── seat_at_table
//      └── serving
//          ├── take_order
//          ├── serve_order
//          └── take_payment

// How do we refer to parts of the module tree?
// ERROR: This wont compile until we declare `hosting` to be public
pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();
    // Relative path
    front_of_house::hosting::add_to_waitlist();
}

// We could also use the `use` kword to brings paths into scope
use crate::front_of_house::hosting::add_to_waitlist;
use crate::front_of_house::serving::{self, take_order, serve_order};  // <- multiple imports
// The `self` symbol is useful to bring a module as well as some of its symbols into scope
pub fn eat_at_restaurant_use_kword() {
    add_to_waitlist();
    take_order();
    serve_order();
    serving::serve_order();
}

// The conventions of what scope to import is pretty similar to the Python idioms
// Similarly, the `as` kword can provide useful aliases
use std::fmt::Result;
pub use std::io::Result as IoResult;  // <-- This can be `use`d in another file

// If you'd like to `use` a symbol that's `used` in another file,
// Just prepend `pub` to the `use` statement in the other file
// (Useful for creating friendly APIs)

// Also, globs! (with obvious warnings about shadowing your symbols)
use std::collections::*;  // Mostly used in writing tests (Ch11)
