use crate::garden::veggies::Asparagus;

// This lines tells the compiler to inclde the code found in `src/garden.rs`
pub mod garden;

fn main() {
    let plant = Asparagus {};
    println!("I'm growing {:?}!", plant);
}
