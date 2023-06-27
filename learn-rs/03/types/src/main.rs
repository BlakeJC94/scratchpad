fn main() {
    // Integer types
    // | Length   | Signed  | Unsigned  |
    // |----------|---------|-----------|
    // | 8-bit    | i8      | u8        |
    // | 16-bit   | i16     | u16       |
    // | 32-bit   | i32     | u32       |
    // | 64-bit   | i64     | u64       |
    // | 128-bit  | i128    | u128      |
    // | arch     | isize   | usize     |
    let x1: isize = 72;
    let x2: u64 = 72;
    println!("Value of x1 is {x1}");
    println!("Value of x2 is {x2}");

    // Floating point types
    // | Length   | Float   |
    // |----------|---------|
    // | 32-bit   | f32     |
    // | 64-bit   | f64     |
    let y1 = 2.0;
    let y2: f32 = 123.45;
    println!("Value of y1 is {y1}");
    println!("Value of y2 is {y2}");

    // Numeric ops
    // addition
    let sum = 5 + 10;
    // subtraction
    let difference = 95.5 - 4.3;
    // multiplication
    let product = 4 * 30;
    // division
    let quotient = 56.7 / 32.2;
    let floored = 2 / 3; // Results in 0
                         // remainder
    let remainder = 43 % 5;

    // Bools
    let t = true;
    let f: bool = false;

    // Chars - single quotes, 4 bytes
    let c = 'z';
    let z: char = 'ℤ'; // with explicit type annotation
    let heart_eyed_cat = '😻';

    // tuples
    // Each position of tuple has seperate type
    let tup: (i32, f64, u8) = (500, 6.4, 1);

    // unpacking works as expected in rust
    let (t1, t2, t3) = tup;
    println!("The value of t2 is {t2}");
    // indexing is done via dot notation (0-based)
    let t2_by_index = tup.1;
    println!("The value of t2_by_index is {t2_by_index}");

    // Arrays, like tuples (except valuesthey all must have the same type)
    let a = [1, 2, 3, 4, 5];
    let b: [i32; 5] = [6, 7, 8, 9, 10];
    // useful when you want stack instead of heap
    // arrays are always of fixed length
    // you can create the array [1, 1, 1, 1] by
    let c = [1; 4];
    // indexing is done by square brackets
    let first = a[0]
    let second = a[1]
}
