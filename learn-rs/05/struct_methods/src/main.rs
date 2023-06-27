fn main() {
    main5();
    main6();
    main7();
}

#[derive(Debug)]
struct Rect {
    width: u32,
    height: u32,
}

impl Rect {
    // `impl` => implementation
    // `self` must be the first parameter
    fn area(&self) -> u32 {
        // `&self` is shorthand for `self: &Self`
        // this method borrows `self`,
        // but methods that take ownership of self can be written
        // (though using `&mut self` is pretty rare in practice)**
        // additionally, methods that borrow self (`&mut self`) mutably can also be written
        self.width * self.height
    }
    // We borrow self here because we don't want to take ownership
    // We just need to read some data from it

    // Note that methods can also share names with fields
    fn width(&self) -> bool {
        self.width > 0
    }
    // The difference is that methods are callables
    // This is a useful way to implement getters,
    // (which aren't automatically implemented)
}
// Everything within the impl block is associated with the `Rect` struct

fn main5() {
    println!("MAIN 5");
    let rect = Rect {
        width: 30,
        height: 50,
    };
    println!("The area of the rect is {:?}", rect.area());

    if rect.width() {
        println!("The rect has non-zero width, it is {}", rect.width);
    } else {
        // NOT REACHED
        println!("The rect has negative width, it is {}", rect.width);
    }
}

// Multiple impl blocks can be defined in the same file? interesting
// Shgould try to keep all this crap toghether though!
impl Rect {
    fn can_hold(&self, other: &Rect) -> bool {
        // `&self` is shorthand for `self: &Self`
        (self.width > other.width) && (self.height > other.height)
    }
}
fn main6() {
    println!("MAIN 6");
    let rect1 = Rect {
        width: 30,
        height: 50,
    };
    let rect2 = Rect {
        width: 10,
        height: 40,
    };
    let rect3 = Rect {
        width: 60,
        height: 45,
    };

    println!("Can rect1 hold rect2? {}", rect1.can_hold(&rect2));
    println!("Can rect1 hold rect3? {}", rect1.can_hold(&rect3));
}



// Associated functions don't have self, generally used for constructors,
// must be accessed with :: instead of .
// You can access the struct with `Self` still
// (It's similar to a classmethod/staticmethod in python?)
impl Rect {
    fn square(size: u32) -> Self {
        // `&self` is shorthand for `self: &Self`
        Self {
            width: size,
            height: size,
        }
    }
}
fn main7() {
    println!("MAIN 7");
    let square = Rect::square(20);
    if square.width == square.height {
        println!("Yep, its a square with size {}", square.width)
    } else {
        println!("uhoh")
    }
}

// **: One use case for `&mut self` is when the method transforms `self`
// and you'd like to prevent the user from using the original object after transform
