// While the compiler is pretty good at preventing issues,
// it's not perfect!

// It's possible to create memory leaks accidentally using Rc and RefCell
// by creating references where items refer to each other in a cycle
// and the reference count doesn't allow the item to be dropped
// (Called a reference cycle)

use crate::List::{Cons, Nil};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

fn main() {
    // ref_cycle();
    weak_ref();
}

// Lets create a cons list
#[derive(Debug)]
enum List {
    Cons(i32, RefCell<Rc<List>>),  // 2nd item in list is a RefCell of Rc
    Nil,
}

// With a tail method to get the end of the list
impl List {
    fn tail(&self) -> Option<&RefCell<Rc<List>>> {
        match self {
            Cons(_, item) => Some(item),
            Nil => None,
        }
    }
}

fn ref_cycle() {
    let a = Rc::new(Cons(5, RefCell::new(Rc::new(Nil))));
    println!("a: initial rc count = {}", Rc::strong_count(&a));
    println!("a: next item = {:?}", a.tail());

    let b = Rc::new(Cons(10, RefCell::new(Rc::clone(&a))));
    println!("a: rc count after b init = {}", Rc::strong_count(&a));
    println!("b: initial rc count = {}", Rc::strong_count(&b));
    println!("b: next item = {:?}", b.tail());

    if let Some(link) = a.tail() {
        *link.borrow_mut() = Rc::clone(&b);
        // dereferencing here to assign to the mutably borrowed value
        // ERROR: overflow?
        // println!("{:?}", link);
    }

    println!("b: rc count after changing a = {}", Rc::strong_count(&b));
    println!("a: rc count after changing a = {}", Rc::strong_count(&a));

    // ERROR: overflow
    println!("a: next item = {:?}", a.tail());
    // If we drop b, RC(b) -> 1 since we cloned &b
    // If we then drop a, the Rc(a) -> 1, since the data of b still refers to it
    // So it is impossible to drop this memory
}

// """
// Creating reference cycles is not easily done, but it’s not impossible either.
// If you have RefCell<T> values that contain Rc<T> values or similar nested
// combinations of types with interior mutability and reference counting,
// you must ensure that you don’t create cycles;
// you can’t rely on Rust to catch them
// """

// But how can we prevent reference cycles? Using Weak RCs can be useful
// Weak Rcs don't express ownership and their count doesn't affect when
// they're cleaned
// (You'll need to upgrade a weakrc before modifying it, but perhaps a small price to pay?)

// For example, let's try to make a tree
#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}
// We want a Node to own its children,
// if a node gets dropped, so should its chilren
// but we also want nodes to have awareness of their parents (w/o owning them)

fn weak_ref() {

    let leaf = Rc::new(Node {
        value: 3,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![]),
    });

    println!(
        "leaf strong = {}, weak = {}",
        Rc::strong_count(&leaf),
        Rc::weak_count(&leaf),
    );
    println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());

    {
        let branch = Rc::new(Node {
            value: 5,
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(vec![Rc::clone(&leaf)]),
        });
        // The node in leaf now has 2 owners

        *leaf.parent.borrow_mut() = Rc::downgrade(&branch);

        println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());
        // There's no infinite output, this didn't crash!

        println!(
            "branch strong = {}, weak = {}",
            Rc::strong_count(&branch),
            Rc::weak_count(&branch),
        );

        println!(
            "leaf strong = {}, weak = {}",
            Rc::strong_count(&leaf),
            Rc::weak_count(&leaf),
        );
    }
    println!("DROP BRANCH");

    println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());
    println!(
        "leaf strong = {}, weak = {}",
        Rc::strong_count(&leaf),
        Rc::weak_count(&leaf),
    );


}
