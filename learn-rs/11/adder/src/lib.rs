// Testing in rust is pretty straightforward (`$ cargo test`)
// - To tests a specific file, `$ cargo test --test <path>`
// - Unit tests tend to be places at the bottom of src files
//     - This enables testing of private interfaces
//     - `$ cargo test --lib`
// - Integration tests are placed in a seperate `tests` directory (next to `src`)
//     - Treats the src module like an extranal lib, no privates avaliable
// - Code snippets in documentation are automatically picked up as well
//     - `$ cargo test --doc`
// - Running  will run the full suite of tests

// `pub` is not needed for getting unit tests working in the same file
fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Debug)]
struct Rect {
    width: u32,
    height: u32,
}
impl Rect {
    fn can_hold(&self, other: &Rect) -> bool {
        other.width < self.width && other.height < self.height
    }
}

// `pub` is needed for external integration tests (see `integration_test.rs`)
pub fn add_two(a: usize) -> usize {
    // ERROR
    // add(a, 3)
    // FIX
    add(a, 2)
}

fn greeting(name: &str) -> String {
    // ERROR
    // String::from("foo")
    // FIX
    format!("Heelo {}!", name)
}

pub struct Guess {
    value: i32,
}

// https://doc.rust-lang.org/rust-by-example/meta/doc.html
impl Guess {
    /// Returns a guess with the value given
    ///
    /// # Arguments
    ///
    /// * `value` - An integer representing a users guess input.
    ///
    /// # Examples
    ///
    /// ```
    /// // You can have rust code between fences inside the comments
    /// // If you pass --test to `rustdoc`, it will even test it for you!
    /// use adder::Guess;
    /// let guess = Guess::new(42);
    /// ```
    pub fn new(value: i32) -> Guess {
        if value < 1 {
            panic!("Guess value must be greater than or equal to 1, got '{}'", value);
        } else if value > 100 {
            panic!("Guess value must be less than or equal to 100, got '{}'", value);
        }
        Guess { value }
    }
}

// This cfg tells compiler to only compile the block when running tests
#[cfg(test)]
mod tests {
    use super::*;  // This is needed in this file because tests is an inner module
    // We'd use items differently if we were writing tests elsewhere

    #[test]
    fn exploration() {
        assert_eq!(2 + 2, 4);
    }

    // #[test]
    // fn another() {
    //     panic!("Force this test to fail")
    // }

    #[test]
    fn larger_can_hold_smaller() {
        let larger = Rect {
            width: 8,
            height: 7,
        };
        let smaller = Rect {
            width: 5,
            height: 1,
        };

        // The `assert!` macro will panic on `false`
        // And also only accepts bools
        assert!(larger.can_hold(&smaller));
    }

    #[test]
    fn smaller_cannot_hold_larger() {
        let larger = Rect {
            width: 8,
            height: 7,
        };
        let smaller = Rect {
            width: 5,
            height: 1,
        };

        // The `assert!` macro will panic on `false`
        // And also only accepts bools
        assert!(!smaller.can_hold(&larger));
    }

    // `assert_eq!` and `assert_ne!` can be used to verify equality/inequality
    #[test]
    fn it_adds_two() {
        assert_eq!(4, add_two(2));
    }
    #[test]
    fn it_doesnt_add_three() {
        assert_ne!(5, add_two(2));
    }
    // Any custom structs or enums will need `#[derive(Debug, PartialOrd)]`
    // to use these macros.
    //
    // Custom failer messages can be added with extras args to the asserts
    #[test]
    fn greeting_contains_name() {
        let name = "Carrol";
        let res = greeting(name);
        assert!(
            res.contains(name),
            "Greeting did not contain name, value was `{}`",
            res,
        );
    }

    // Use `should_panic` to verify expected failures
    // Specfic panics can be tested for as well
    #[test]
    // #[should_panic]
    #[should_panic(expected = "less than or equal to 100")]
    fn greater_than_100() {
        Guess::new(200);
    }

    // It's also possible tow write tests that return a Result (instead of shitting the bed)
    #[test]
    fn it_works_with_result() -> Result<(), String> {
        if 2 + 2 == 4 {
            Ok(())
        } else {
            Err(String::from("2+2!=4"))
        }
    }
    // [should_panic] can also be covered here with `assert(value.is_err())`
}

