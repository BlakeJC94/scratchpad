use traitobj::{Button, Screen, Draw};

// Now a user can create their own object that implements the trait

struct SelectBox {
    width: u32,
    height: u32,
    options: Vec<String>,
}

impl Draw for SelectBox {
    fn draw(&self) {
        println!("Code to draw selectbox goes here!");
    }
}

fn main() {

    let screen = Screen {
        components: vec![
            Box::new(SelectBox {
                width:75,
                height:10,
                options: vec![
                    String::from("foo"),
                    String::from("bar"),
                    String::from("baz"),
                ],
            }),
            Box::new(Button {
                width:50,
                height:20,
                label: String::from("wumbo"),
            }),
        ]
    };
    screen.run();

}


// The only cost of this approach is dynamic dispatch
// The compiler will use pointers to methods to compile the code
// since it cant "inline" every possible implementation
// which comes with a small performance cost
// but our code was much more flexible and easy to write!
