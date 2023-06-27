// Traits with associated types connect a type placeholder
// with a trait such that the trait method definitions
// can use these placeholder types in the signatures

// For example, this is what the stdlib trait `Iterator` kinda looks like:
pub trait Iterator {
    type Item;  // `Item` is a placeholder type in the trait implementation ..
    fn next(&mut self) -> Option<Self::Item>;  // .. Which is used here
}
// Implementors of this trait will need to give a concrete type

struct Counter { }  // [placeholder]
impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        Some(5)  // [placeholder]
    }
}

// Why not just use a generic type?
// pub trait Iterator<T> {
//    fn next(&mut self) -> Option<T>;
// }
// Because then we'd have to annotate the types in each implementation

// Associated types become part of the contract
// And will require defining the stand-in types



fn main() {
    println!("Hello, world!");
}
