use std::ops::Deref;

fn main() {
    main1();
    main2();
    main3();
    main4();
}

// Using references
fn main1() {
    println!("MAIN1");
    let x = 5;  // x holds a value 5
    let y = &x;  // y is a reference, pointing to the value that x holds
    assert_eq!(5, x);
    assert_eq!(5, *y);  // We use * to follow the pointer (de-reference it)
}

// Using boxes
fn main2() {
    println!("MAIN2");
    let x = 5;  // x holds a value 5
    let y = Box::new(x);  // y is a box, pointing to the value that x holds
    assert_eq!(5, x);
    assert_eq!(5, *y);  // We use * to follow the pointer (de-reference it)
}

// Lets build our own smart pointer!
struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}

// Recall that so implement a trait, we need to implement the requirement methods from that trait
impl<T> Deref for MyBox<T> {
    type Target = T;  // (This is covered more in Ch19)
    // Borrow self and return a ref
    fn deref(&self) -> &Self::Target{
        &self.0
    }
    // When * is called on something that's not a reference,
    // rust actually does `*(y.deref())` instead!
}

fn main3() {
    println!("MAIN3");
    let x = 5;  // x holds a value 5
    let y = MyBox::new(x);  // y is a mybox, pointing to the value that x holds
    assert_eq!(5, x);
    assert_eq!(5, *y);  // We use * to follow the pointer (de-reference it)
}

fn yo(name: &str) {
    println!("Yo, {name}!");
}

fn main4() {
    println!("MAIN4");
    let m = MyBox::new(String::from("Rust"));
    yo(&m); // Wait, whats the deal here?
    // Shouldn't this error out like this?
    // yo(String::from("foo"));
    // This works because of "deref conversion"

    // Because we have `deref` on `MyBox`,
    // `&MyBox` is converted to `&String`.
    // But rust calls `deref` again to convert
    // `&String` to `&str`

    // Rust will deref as many times as it can in order to match the parameters type
    // Without this, we'd have to write
    // `yo(&(*m)[..])`, which is much harder to read!
}

// The trait `Deref` overrides the `*` operator on immutable refs
// The trait `DerefMut` overrides the `*` operator on mutable refs
//
// """
// Rust does deref coercion when it finds types and trait implementations in three cases:
//
//     From &T to &U when T: Deref<Target=U>
//     From &mut T to &mut U when T: DerefMut<Target=U>
//     From &mut T to &U when T: Deref<Target=U>
// """
//
// Note that it's still not possible to change an immutable ref into a mutable one
