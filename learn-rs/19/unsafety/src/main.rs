use std::slice;
// Unsafe rust can be used to do lower-level operations such as
// - re-referenceing a raw pointer
// - calling unsafe functions/methods
// - modify mutable static variables
// - implement unsafe traits
// - access fields of `unions`s

// The borrow checker is still active, other safety checks are still active

// Keep unsafe blocks as small as possible, future debuggers will be thankful!

// We now have _raw pointers_ to use:
// - allowed to have multiple raw pointers for the same location
// - aren't guaranteed to point to valid memory
// - allowed to be null
// - don't get automatically cleaned up

fn main() {
    raw_pointers();
    // ERROR
    // dangerous();
    // FIX
    unsafe { dangerous() };
    split_at_mut_example();


    let mut v = vec![1, 2, 3, 4, 5, 6];
    let (a, b) = my_split_at_mut(&mut v[..], 3);
    assert_eq!(a, &mut [1, 2, 3]);
    assert_eq!(b, &mut [4, 5, 6]);

    using_c_abs();

    unsafe { println!("{}", COUNTER) }
    add_to_counter(3);  // Doing this across threads would probably result in data races
    unsafe { println!("{}", COUNTER) }
}

fn raw_pointers() {
    let mut num = 5;

    // We can create raw pointers safely..
    let r1 = &num as *const i32;  // immutable raw pointer
    let r2 = &mut num as *mut i32;  // mutable raw pointer

    // .. but we can't dereference them safely
    // ERROR
    // println!("r1 is: {}, r2 is: {}", *r1, *r2);
    // FIX
    unsafe {
        println!("r1 is: {}, r2 is: {}", *r1, *r2);
    }

    // Creating raw points at arbitrary addresses is fine
    let address = 0x012345usize;
    let r = address as *const i32;

    // But reading them is unsafe
    // ERROR: segfault lol
    // unsafe {
    //     println!("What's at address '{}'? It's a '{}'!", address, *r);
    // }
}

// Why in the hell would we give a shit about raw pointers?
// Usually when we're interfacing with C code

unsafe fn dangerous() {
    println!("This is an unsafe function! wooo!");
    let mut num = 10;

    let r1 = &num as *const i32;  // immutable raw pointer
    let r2 = &mut num as *mut i32;  // mutable raw pointer
    // Look Ma! No unsafe block needed in an unsafe function!
    println!("r1 is: {}, r2 is: {}", *r1, *r2);
}

// Say we want to create a function that splits a mutable vector into two vectors at a given index
// This isn't possible with safe rust,
// so we'll use unsafe rust and create a safe abstraction

fn split_at_mut_example() {
    let mut v = vec![1, 2, 3, 4, 5, 6];

    let r = &mut v[..];

    let (a, b) = r.split_at_mut(3);

    assert_eq!(a, &mut [1, 2, 3]);
    assert_eq!(b, &mut [4, 5, 6]);
}

// Note that this function isn't marked as unsafe, despite using an unsafe block
fn my_split_at_mut(values: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = values.len();
    assert!(mid < len);
    // ERROR second mutable borrow -- verboten!
    // (&mut values[..mid], &mut values[mid..])
    // FIX
    let ptr = values.as_mut_ptr();
    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

// Lets call some C functions as well!
// The input to `extern` is which aplication binary interface (ABI) to use
extern "C" {
    fn abs(input: i32) -> i32;
}

fn using_c_abs() {
    unsafe {
        println!("Absolute value of -3 according to C: {}", abs(-3));
    }
}

// We can also create functions in rust that can be called from C!
#[no_mangle]  // This stops the compiler from changing the name
pub extern "C" fn call_from_c() {
    println!("Rusty function called in C!");
}  // No unsafe required, this is C's problem to deal with!


// Unsafe cal also be used to mess with global variables
static mut COUNTER: u32 = 0;

fn add_to_counter(inc: u32) {
    unsafe { COUNTER += inc; }
}

