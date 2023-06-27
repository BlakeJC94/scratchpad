// Most cases, a value has one clear owner
// But what about cases where a value has multiple owners (such as a graph?)

// the stdlib Rc smart pointer is useful to count these
// Values will only be dropped from an Rc pointed when there's no owners


fn main() {
    main1();
    main2();
}

enum List {
    Cons(i32, Box<List>),
    Nil,
}

use std::rc::Rc;

use crate::List::{Cons, Nil};

fn main1() {
    println!("MAIN1");
    let a = Cons(5, Box::new(Cons(10, Box::new(Nil))));
    // ERROR:
    // let b = Cons(3, Box::new(a));
    // let c = Cons(4, Box::new(a));
    // We can't have t symbols owning the same data like that
    // The value is moved to b, and thus cannot be used in c

    // We could change Cons to hold references instead of values,
    // But we would need to litter lifetimes all
    // over our code to make that work,
    // which is not always appropriate

    // But, if we use Rc, we can sidestep ownership of values
}

enum RcList {
    RcCons(i32, Rc<RcList>),
    RcNil,
}

use crate::RcList::{RcCons, RcNil};

fn main2() {
    println!("MAIN2");

    let a = Rc::new(RcCons(5, Rc::new(RcCons(10, Rc::new(RcNil)))));
    println!("count after creating a = {}", Rc::strong_count(&a));

    let b = RcCons(3, Rc::clone(&a));
    println!("count after creating b = {}", Rc::strong_count(&a));

    {
        let c = RcCons(4, Rc::clone(&a));
        println!("count after creating c = {}", Rc::strong_count(&a));
    }
    println!("count after dropping c = {}", Rc::strong_count(&a));
    // Every time we call Rc::clone, the reference count is incremented by 1
    // And each time drop is called on and Rc, the count decreases by 1
}

// Should go without saying at this point,
// But no mutability is allowed with this system for obvious reasons
