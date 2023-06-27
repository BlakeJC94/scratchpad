fn main() {
    // References are like pointers,
    // theyre addresses that can be followed to access data of a variable
    // Prefix & to a variable to specify a reference
    let s1 = String::from("hello");
    let len = calculate_len(&s1);
    println!("The length of '{}' is {}.", s1, len);
    // References allow us to refer to a value
    // without taking ownership of the variable

    // Note, de-referencing is done by the * prefix, which will be explored later

    // Creating a reference is referred to as "borrowing"

    // Mutation of a borrowed value wont work though
    // ERROR
    // change(s);

    // Refernces are immutable by default, but can be mutable if asked nicely
    // in the function signature and refernce declaration, and function call
    let mut s = String::from("wumbo");
    println!("`s` was {}", s);
    change(&mut s);
    println!("`s` is now {}", s);

    // The man caveat with refernces:
    // Only ONE mutable refernce may exist for a value
    // You can't borrow what's already been borrowed!
    // (the upshot is that Rust will refuse to compile code with potential race conditions)

    let mut s = String::from("hello");
    let r1 = &mut s;
    // ERROR
    // let r2 = &mut s;
    // println!("{}, {}", r1, r2);

    // Braces can be used to control the scope in this sitation
    // (but still can't be used at the same time)
    {
        let r1 = &mut s;
    }
    let r2 = &mut s;

    // For obvious reasons,
    // you can't have a refernce and a nother mutable refernce to th esame values
    // Unless.. the immutalbe reference is out of scope (or not used again)


    // What about dangling references?
    // What if we have a reference to a value that has droppped its data due to scope??
    // ERROR
    // let ref_to_nothin = dangle();
    // FIXED
    let transferred_ownership_so_not_deallocated_on_function_end = dangle();
}

fn calculate_len(s: &String) -> usize {
    // `s` points to `s1` (which itself points to some text data)
    s.len()
    // After s is out of scope, there's no values to drop
    // Its basically like a simple type!
}

// ERROR
// fn change(s: &String) {
//     s.push_str("floob");
// }
fn change(s: &mut String) {
    s.push_str("floob");
}

//ERROR: some shit about lifetimes?
// fn dangle() -> &String {
//     let s = String::from("yo");
//     &s
// } // s is out of scope, so the data get dropped. Rust wont comile due to the dangling ref!
// FIXED
fn dangle() -> String {
    let s = String::from("yo");
    s  // Transfer ownership of data to caller, all is well
} // s is out of scope, so the data get dropped. Rust wont comile due to the dangling ref!
