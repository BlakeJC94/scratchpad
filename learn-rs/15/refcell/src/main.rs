// Making a mutable ref to mutable data is usually verboten by rust
// But sometimes it's needed, and can be acceptable if it's wrapped around a safe API
// (The compiler will need some convincing though)

// This sort of pattern would come up, for example, when a value needs to mutate itself
// but appear immutable to other code (such as a mock object)

fn main() {
    let _x = 5;
    // ERROR:
    // let y = &mut x;
}

// Check out lib.rs for a practical example
