// Keep in mind that `mod` != "include" from other langs
// `mod` just declares the module tree to the compiler
mod front_of_house;
// This will assign all the code in `src/front_of_house.rs` to
// `crate::front_of_house`, which can then be `use`d

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
