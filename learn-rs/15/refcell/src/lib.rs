// Lets make a lib that tracts a value against a maximum value
// and sends a message based on how close the value is to max

// How could we test this using a mock?

// Lets declare a trait
pub trait Messenger {
    fn send(&self, msg: &str);
}

// And define a struct with a generic type param
// that is limited to types with the `Messenger` trait
// and has a declared lifetime at least as long at the reference it holds
pub struct LimitTracker<'a, T: Messenger> {
    messenger: &'a T,
    value: usize,
    max: usize,
}

// And implement a constructor and setter for this struct
impl<'a, T> LimitTracker<'a, T>
where
    T: Messenger,
{
    pub fn new(messenger: &'a T, max: usize) -> LimitTracker<'a, T> {
        LimitTracker {
            messenger,
            value: 0,
            max,
        }
    }

    // Note that a mutable reference to self is required here
    pub fn set_value(&mut self, value: usize) {
        self.value = value;

        let percentage_of_max = self.value as f64 / self.max as f64;

        if percentage_of_max >= 1.0 {
            self.messenger.send("Error: exceeded quota!");
        } else if percentage_of_max >= 0.9 {
            self.messenger.send("Warning: over 90% of quota!");
        } else if percentage_of_max >= 0.75 {
            self.messenger.send("Info: over 75% of quota!");
        }
    }
}

// Cooool, lets try to make a mock version of this for testing

#[cfg(test)]
mod tests {
    use super::*;

    // struct MockMessenger {
    //     sent_messages: Vec<String>,
    // }

    // impl MockMessenger {
    //     fn new() -> MockMessenger {
    //         MockMessenger {
    //             sent_messages: vec![],
    //         }
    //     }
    // }

    // Lets implement the trait `Messenger` for `MockMessenger`
    // impl Messenger for MockMessenger {
    //     fn send(&self, message: &str) {
    //         self.sent_messages.push(String::from(message));
    //     }
    // }
    // We can't modify MockMessenger to keep track of messages,
    // since the `send` method takes and immutable ref to self
    // (If we make change `&self` to `&mut self`, we don't match the sig)


    // #[test]
    // fn it_sends_an_over_75_percent_info_message() {
    //     let mock_messenger = MockMessenger::new();
    //     let mut limit_tracker = LimitTracker::new(&mock_messenger, 100);

    //     // This should trigger the warning message
    //     limit_tracker.set_value(80);

    //     assert_eq!(mock_messenger.sent_messages.len(), 1);
    // }

    // Lets use a RefCell in the mock messenger and wrap sent_messages
    use std::cell::RefCell;
    struct MockMessenger {
        sent_messages: RefCell<Vec<String>>,
    }

    impl MockMessenger {
        fn new() -> MockMessenger {
            MockMessenger {
                sent_messages: RefCell::new(vec![]),
            }
        }
    }

    impl Messenger for MockMessenger {
        fn send(&self, message: &str) {
            self.sent_messages.borrow_mut().push(String::from(message));
        }

    }

    #[test]
    fn it_sends_an_over_75_percent_info_message() {
        let mock_messenger = MockMessenger::new();
        let mut limit_tracker = LimitTracker::new(&mock_messenger, 100);

        // This should trigger the warning message
        limit_tracker.set_value(80);

        assert_eq!(mock_messenger.sent_messages.borrow().len(), 1);
    }
}

// We've just used RefCell to mutate an inner value
// while keeping the outer value immutable
