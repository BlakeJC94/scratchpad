// Say we want to make an `OutlinePrint` trait with an `outline_print` method

use std::fmt;

trait OutlinePrint: fmt::Display { // (Similar to specifying trait bounds)
    fn outline_print(&self) {
        let output = self.to_string();  // to_string is already there for any type with `Display`
        let len = output.len();
        println!("{}", "*".repeat(len + 4));
        println!("*{}*", " ".repeat(len + 2));
        println!("* {} *", output);
        println!("*{}*", " ".repeat(len + 2));
        println!("{}", "*".repeat(len + 4));
    }
}

// #[derive(Debug)]
struct Point
{
    x: i32,
    y: i32,
}

// ERROR:
// impl OutlinePrint for Point {}
// FIX:
impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
impl OutlinePrint for Point {}

fn main() {
    let p = Point {x: 3, y: 4};
    p.outline_print();

}
