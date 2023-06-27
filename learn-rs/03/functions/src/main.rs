fn main() {
    println!("Hello, world!");
    another_function(5, 'h');
    example_of_expression();
    let five = example_of_return_value();
    println!("value of five is {five}");
    let six = example_of_transform(five);
    println!("value of six is {six}");
}

// Parameters
// Annotations must be specified for args
// Different params seperated by commas
fn another_function(x: i32, unit_label: char) {
    println!("Another function!");
    println!("The measurement is {x}{unit_label}.");
}

fn example_of_expression() {
    let y = {
        let x = 3;
        x + 1
    };  // expression in {..} outputs 4
    // Note the lack of semicolon on last line
    // If a semicolo is placed there, it becomes a statement and returns nothing
    println!("The value of y is {y}")
}

// Types of return must be specified in function header
fn example_of_return_value() -> i32 {
    5
}

// Same deal as with `example_of_expression`,
// putting a semicolon after the end of the last statement
// makes the function return nothing
fn example_of_transform(x: i32) -> i32 {
    x + 1
}
