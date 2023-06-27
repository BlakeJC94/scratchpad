// "String" may refer to the `str` literal type, or the stdlib `String` collection type
// str slices are static, whereas String is more like a vector
// (I wonder if theres a duality between vec/array anf str/String?)
fn main() {
    // Creating a new String, where we can push data into throughout the program
    let mut s1 = String::new();
    println!("New (empty) string: {s1}");

    // Or I could create one from data
    let data = "initial data";
    let s2 = data.to_string();  // Method could also be called on literal directly
    // Or, as we're seen, could aslo be done with the `String::from(..)` constructor
    println!("Another new string (from data): {s2}");

    // Updating a string is easy as well
    s1.push_str("pushed data");
    s1.push('.');  // Push can be used to add a single char
    s1.push_str("more pushed data");
    println!("Added data to string: {s1}");

    // Combining 2 strings can be done with +
    let s3 = s2 + &s1;  // <- s2 has been moved after this, no longer in scope
    println!("Combined the strings: {s3}");
    // println!("Error if accessing moved value: {s2}");

    // This notation (and borrowing shit) can get confusing quickly with > 2 variables
    let s1 = String::from("tic");
    let s2 = String::from("tac");
    let s3 = String::from("toe");
    // let s = s1 + "-" + &s2 + "-" + &s3;
    // use format! macro instead
    let s = format!("{}-{}-{}", s1, s2, s3); // Similar to `println!`
    println!("concatted string: {s}");

    // Does indexing into strings work?
    // In short, no
    // This is because strings are basically wrappers around Vec<u8>
    // Each char is 2 bytes, so indexing elem 0 of "hello" would return 104, not 'h'
    // .. But you can slice strings
    let s_slice = &s[0..4];  // idxs are bytes, not chars!
    println!("slice of s: {s_slice}");
    // Careful not to slice a character in half though
    let s4 = String::from("Здравствуйте");
    // let s_slice = &s4[0..1]; // PANIC:
    // println!("slice of s: {s_slice}");

    // Can we iterate over strings?
    // Sure, but you need to be explicit about bytes or chars
    println!("printing_chars");
    for c in s4.chars() {
        print!("{c}, ")
    }
    println!("");
    println!("printing_bytes");
    for c in s4.bytes() {
        print!("{c}, ")
    }
    println!("");

}
