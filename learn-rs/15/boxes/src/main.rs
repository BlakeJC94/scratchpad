fn main() {
    // Here's how to store an i32 on the heap:
    let b = Box::new(5);
    println!("b = {:?}", b);  // print doesn't mention box?

}

// Single values on the heap like this are pretty pointless (heh)
// But this pattern allows for recursive types!

// E.g. the "cons list" (1, (2, (3, Nil)))
// Each item of the list has 2 elements, the value and the next item

// ERROR:
// enum List {
//     Cons(i32, List),
//     Nil,
// }
// FIX:
enum List {
    Cons(i32, Box<List>),
    Nil,
}

// The reason we need a Box here is because the compiler would try to estimate the size needed for
// the type if we didn't provide it, but the amount of space for a single list is seemingly infinite!

// Instead of storing the value directly, we can indirectly store it with a pointer, which is
// always a fixed size (and very finite)


use crate::List::{Cons, Nil};
fn main1() {
    println!("MAIN1");
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Nil)))))
    //Phew that's a lot of boxes, but it breaks the recursion error in psace allocation
}

// Boxes are a simple example of "Smart pointers" because they implement
// the "Deref" and "Drop" traits, which allows Boxes to be treated like
// refs (out of scope => out of access)
