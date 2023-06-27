// (Using the same enum as before)
#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
}
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),  // <-- This variant holds data
    Button,
}

// if and let can be combined to simplyfy a match statement
fn main() {
    println!("Hello, world!");
    let config_max = Some(3u8);

    match config_max {
        Some(max) => println!("The maximum is {}", max),
        _ => (),
    }

    // if let makes the symbol `max` bind to `config_max`
    // when `config_max` is a `Some` variant
    if let Some(max) = config_max {
        println!("The maximum is {}", max)
    }
    // Less indentation, less boilerplate code

    // An `else` statement can also be included
    let coin = Coin::Quarter(UsState::Alaska);
    let mut count = 0;
    if let Coin::Quarter(state) = coin {
        println!("State quarter from {:?}!", state);
    } else {
        count += 1;
    }
    println!("Count = {}", count);

}
