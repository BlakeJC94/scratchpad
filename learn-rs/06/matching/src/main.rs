// `match` is a vary useful "coin sorting" construct,
// particularly with handling `Option`s, or any other kind of enum
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

fn main() {
    let mut sum: u8 = 0;
    let coins: [Coin; 5] = [
        Coin::Nickel,
        Coin::Dime,
        Coin::Quarter(UsState::Alaska),
        Coin::Penny,
        Coin::Button,
    ];
    println!("I've got 4 coins, let's add them up..");
    for coin in coins {
        sum += value_in_cents(coin);
    }
    println!(
        "I added up the value of 4 coins in a loop, the result was {}",
        sum
    )
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        // Interestingly, the compiler will complain if not all cases are covered
        Coin::Penny => {
            println!("Woo! Lucky Penny!");
            1
        },
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            // Match arms can also bind to parts of values as well!
            println!("State quarter from {:?}!", state);
            25
        },
        other => {
            // I can use the value `other` in this branch
            // Or if not needed, I can replace `other` with `_`
            println!("I don't think this is a coin..");
            0
        },
    }
    // If each arm is a function call, we can replace an arm with an "empty" function call
    // to say nothing will happen (by using `value => ()`, the empty tuple)
}

