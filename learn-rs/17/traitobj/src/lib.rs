// Lets make a tiny gui library with "Component objects" (such as
// button, listview, etc.) that inherit from a common base "object"

// Rust doens't have an inheritance system like other languages,
// but this pattern is still possible!

pub trait Draw {
    fn draw(&self);
}

// Define a Screen struct with a trait object
pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,
    // Recall that `Box` allows us to point to data on the heap.
    // `dyn Draw` is a stand-in for "type that implements the `Draw` trait"
}

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}


// Using a generic type seems like a reasonable approach to this:
// pub struct Screen<T: Draw> {
    // pub components: Vec<T>,
// }

// impl<T> Screen<T>
// where
    // T: Draw,
// {
    // pub fn run(&self) {
    //     for component in self.components.iter() {
    //         component.draw();
    //     }
    // }
// }
// BUT! This will only allow homogenous types in the vector,
// Which means the vec need to have all the same components

// Lets create a button struct and implement the Draw trait
pub struct Button {
    pub width: u32,
    pub height: u32,
    pub label: String,
}

impl Draw for Button {
    fn draw(&self) {
        println!("Code to draw button goes here!")
    }
}
