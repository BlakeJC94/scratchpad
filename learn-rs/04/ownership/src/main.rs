fn main() {

    // s not defined yet
    {
        // s not defined yet
        let s = "hello";
        // s is valid
        println!("{s}");
    }
    // s is now out of scope

    // All the basic types are a known size and can easily be stored on the stack
    // Strings are slightly more complex, they need to be on the heap
    let mut s = String::from("wumbo");
    // (~malloc) This `from` method requests memory from the heap
    // (~free) Once the var is out of its own scope, the memory is freed automatically
    // this is done automatically with a rs fn called `drop`
    println!("{s}");

    // Basic hardcoded literals are superfast and useful,
    // but not suitable for user input
    // These strings can be mutated
    s.push_str(", floob");
    println!("{}", s);

    println!("Move some data about");
    let mut x = 5;
    let y = x;
    println!("x = {x}, y = {y}");
    // These are simple ints, allocated to the stack and copied when y is bound to x

    // What about strings of the heap?
    let s1 = String::from("wumbo");
    println!("s1 = {s1}");
    let s2 = s1;
    println!("s2 = {s2}, s1 out of scope");
    // Looks similar, but slightly different
    // A string is made of a (pointer, bytes_len, capacity), simple stck vars
    // binding s2 to s1 copies this tuple,
    // but note the DATA pointed to is NOT COPIED!

    // Also, this binding makes s1 reduntant
    // So to prevent potential "double free" memory bugs
    // s1 is automatically deallocated
    // Because (I think) s2 now *owns* the heap value that s1 had

    // ERROR: `borrow of moved value`
    // println!("{}, world!", s1);

    // Note this is shallow copy + de-allocation is called a "move"
    // Deep copying must always be done manually,
    // So simple rust code is always the fastest option
    // A deep copy can be done with string.
    let s3 = s2.clone();
    println!("s2 = {s2}, s3 = {s3}");

    println!("Make s2 mutable, copy again and try to mutate s2");
    let mut s2 = s2;
    let s3 = s2.clone();
    s2.push_str(", mcgumbo");
    println!("s2 = {s2}, s3 = {s3}");
    // No changes made to s3, it was a deep copy across the heap!

    // So why was this so simple for integers?
    // x and y have known sizes at compile time, so no heap is needed
    // Does y check if x changes?
    println!("Adding 1 to x");
    x = x + 1;
    println!("x = {x}, y = {y}");
    // IT DOESNT! so this is a full copy on the stack

    // Writing functions works with this same concept
    // Passing a variable to a function will move or copy,
    // (depending on the context)
    println!("Function example");
    let s = String::from("foobar");
    takes_ownership(s);
    println!("`s` no longer valid after func, since heap val was owned by the func param");
    let x = 5;
    makes_copy(x);
    println!("`x` = {} is valid after func, since copy was used on the stack", x);

    // Functions can also transfer ownership
    let v1 = gives_ownership();
    let v2 = String::from("hello");
    let (v3, v4) = takes_and_gives_back(v2);

    println!("====");
    println!("KEY TAKAWAYS:");
    println!("  * assigning a value to another variable always moves it");
    println!("  * values with heap data are cleared after moving out of scope");
}

fn takes_ownership(some_string: String) {  // some_string in scope
    println!("some_string: {}", some_string);
} // some_string out of scope, `drop` is called automatically to free heap

fn makes_copy(some_int: i32) { // some_int in scopt
    println!("some_int: {}", some_int);
}

fn gives_ownership() -> String {
    let val = String::from("yours");
    val  // Without a semi-colon, this function returns the value (and ownership)
}

fn takes_and_gives_back(val: String) -> (usize, String) {
    (val.len(), val)  // Note that (val, val.len()) wont work, dealloc is fast!!
}

// Lets look at references next time, which make things a bit easier
