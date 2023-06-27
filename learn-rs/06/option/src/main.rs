// The `Option` enum is so dang useful, Rust doesn't need a "null" type!
// - Prevents hidden bugs where a symbol is accidentally set to "null"
//   and non-null methods are called on it
// - Forces rust code to explicitly handle nulls if they are a possibility

// It's so dang useful, it doesn't even need to be imported to the namespace
// It looks like this in stdlib:
// ```rust
// enum Option<T> {
//     None,
//     Some(T),
// }
// ```

// Wait, whats the `<T>` syntax??
// Not covered yet (see Ch10), but it's a "generic type parameter"
// - Allows the `Some` variant of the `Option` to hold
//   one piece of data of type `T`

fn main() {
    println!("Hello, world!");
    let some_number = Some(5);
    let some_char = Some('e');

    // Generic `None` is verboten, only declared types are allowed by compiler
    let absent_number: Option<i32> = None;

    // Here's an example of why `Option` is useful:
    let x: i8 = 5;
    let y: Option<i8> = Some(5);
    let z: Option<i8> = None;

    // ERROR: trying to add i8 to Option<i8>
    // let sum = x + y;
    // FIX:
    let sum_xy = match &y {
        Some(y) => x + y,
        _ => 0,
    };
    println!("Adding {} and y, where y is Some(5), results in {}", x, sum_xy);

    let sum_xz: i8 = match &z {  // type not needed, but it is polite!
        Some(z) => x + z,
        None => -1,
    };
    println!("Adding {} and z, where z is None, results in {}", x, sum_xz);
    println!("(because we explicitly told it to be!)");
}
// We will see more of this in the next section
