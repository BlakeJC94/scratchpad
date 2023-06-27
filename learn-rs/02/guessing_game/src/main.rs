use std::io;  // bring io module from stdlib into scope
use rand::Rng;  // Get the trait implemented by random number generators
use std::cmp::Ordering;  // `Ordering` is another enum train with Less/Greater variants

fn main() {
    println!("Guess the number!");

    // Use a thread-local `Rng` and call `get_range`
    // which takes an expression as an arg (start..=end)
    // inclusive of bounds
    let secret_number = rand::thread_rng().gen_range(1..=100);
    // println!("TRACE The scret number is {secret_number}");
    // Final touch: remove debugging statements

    // How do you know which traits to get and which methods to use??
    // `$ cargo doc --open` will open the docs for all dependencies in a browser

    // Start an infinite loop
    loop {

        println!("Please input your guess.");

        // Create a variable which is mutable
        // `let` is the keyword used to create variables
        // All variables are immutable by default, use `mut` to make them mutable
        // `String` is a type of growable utf-8 encoded bit of text
        // `::` is used to access associated functions with types
        let mut guess = String::new();

        io::stdin()  // Returns instance of `std::io::Stdin`
            .read_line(&mut guess)  // read stdin into address of mutable `guess`
            .expect("Failed to read line");  // Handle potential failures
        // `Stdin.read_line(..)` returns a `Result` instance, if it's an `Err`
        // `Result.expect(..)` will simply crash the program if not `Ok`

        // Match expression with multiple arms
        // `guess.cmp` will return a `Ordering` variant
        // match guess.cmp(&secret_number) {
        //     Ordering::Less => println!("Too small!"),
        //     Ordering::Greater => println!("Too big!"),
        //     Ordering::Equal => println!("Bingo!"),
        // }
        // But this didnt compile, with the following error:
        // ```
        // = note: expected reference `&String`
        //            found reference `&{integer}`
        //
        // ```
        // `secret_number` is a number, not a string
        // let's parse the `guess` string into an unsigned 32bit integer
        // let guess: u32 = guess.trim().parse().expect("Please type a number!");
        // `String::parse` will use the type annotation specified (after `:` of variable name)
        // `match` staements can also return values
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed : {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("Bingo!");
                break;  // break the loop only when correct
            }
        }
    }

    // Add "rand" = 0.8.3 as a dependency in Cargo.toml
}
