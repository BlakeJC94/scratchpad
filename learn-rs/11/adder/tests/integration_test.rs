use adder;

mod common;

#[test]
fn it_adds_two() {
    assert_eq!(4, adder::add_two(2));
}

#[test]
fn it_adds_two_with_commons() {
    common::setup();
    assert_eq!(common::get_four(), adder::add_two(2));
}

// Note that code in `main.rs` can't be used in integration tests
// This is the main reason why the core rust code is placed in `lib.rs`
// Ideally, the `main.rs` is simple enough that there are obviously no deficiencies
