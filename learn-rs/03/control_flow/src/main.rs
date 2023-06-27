fn main() {
    let number = 3;

    // If statements are normal
    // Though it should be noted that conditions must evaluate to a bool
    if number < 5 {
        println!("Condition was true")
    } else {
        println!("Condition was false")
    }

    // else ifs are also straightforward
    if number < 5 {
        println!("number is less than 5")
    } else if number < 10 {
        println!("number is less than 10")
    } else {
        println!("number is at least 10")
    }

    // ternary statements are also straightforward
    let condition = true;
    let number = if condition { 5 } else { 6 };
    println!("The value of number is {number}");
    // Though, this should output the same type
    // for variable binding reasons

    // The loop kword is used for "while true" statements
    // As seen in 02, the break kword can be used to exit a loop
    let mut i = 0;  // Note that this break wont work if we used let i = 0; ... let i = i + 1
    println!("tsarting a loop");
    loop {
        i = i + 1;
        println!("i = {i}");
        if i > 5 {
            println!("exiting a loop");
            break
        }
    }

    // Values can be returned directly from loops too
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;
        }
    };
    println!("The result is {result}");

    // Also what? you can specify loop labels with a single quote??
    let mut count = 0;
    'counting_up: loop {
        println!("count = {count}");
        let mut remaining = 10;

        loop {
            println!("remaining = {remaining}");
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }
            remaining -= 1;
        }
        count += 1;
    }
    println!("End count = {count}");

    // using loop/if/break is the long way of using a while loop
    let mut number = 3;
    while number != 0 {
        println!("  {number}");
        number -= 1;
    }
    println!("  liftofff!");

    // Instead of using a while/if/index variable,
    // use a for loop to iterate through a collection
    let a = [10, 20, 30, 40, 50];
    for e in a {
        println!("The value is {e}");
    }

    // stdlib has plenty of useful necessities for this kind of stuff
    for number in (1..4).rev() {
        println!("{number}!")
    }

    ex_factorial();
}

fn ex_factorial() {
    let n = 7;
    let factorial = fact(n);
    println!("The value of {n}! is {factorial}");
}

fn fact(n: i32) -> i32 {
    if n <= 1 {
        1
    } else {
        n * fact(n - 1)
    }
}
