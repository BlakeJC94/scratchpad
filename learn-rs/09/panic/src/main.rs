use std::{fs::File, io::{ErrorKind, self, Read}};

fn main() {
    // How to cause a panic:
    // ERROR:
    // panic!("Hello, world!");

    // Using backtrace is easy, run program with RUST_BACKTRACE=1
    // ERROR:
    // let v = vec![1, 2, 3];
    // v[99];

    // Some errors are recoverable though, and aren't worth shitting the bed over
    // Such as when a file doesn't exist.
    // This is where the `Result<T, E>` enum, like `Option`, is verrrrry useful
    let file_result = File::open("foo.txt");
    println!("result = {:?}", file_result);

    // If the file doesn't exist, this is an Err(..).
    // We could panic when an Err is found
    // Or, say we would like to cerate the file first in that case.
    // We can use an inner match statement to match on a kind of error
    let file = match file_result {
        Ok(file) => file,
        // Err(error) => panic!("Ahhh no find of the file! {:?}", error),
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("foo.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating file? {:?}", e),
            },
            other_error => {
                panic!("Problem opening file {:?}", other_error);
            },
        }
    };

    // But yikes, 3 nested matches? This was hard to write and debug
    // In ch13, we'll learn about closures, which can simplify this dramatically

    // Before then, we'll learn about some shortcuts: `unwrap` and `expect`
    // ERROR:
    // let file2 = File::open("bar.txt").unwrap();
    // This will panic if there's an error,
    // meaning we don't need to write a trivial match statement
    // `expect` gives a slightly nicer panic message..


    // Match statements can be used to propegate and act on recoverable errors
    // ERROR:
    // let username = match read_username_from_file_1() {
    //     Ok(user) => user,
    //     Err(e) => panic!("Ahhh {:?}", e),
    // };
    // There's a shorter way to implement this logic without writing a bunch
    // of match statements inside the function (using `?`, which will unpack Ok values and early
    // return Err values)




    // let file = match file_result {
    //     Ok(file) => file,
    //     Err(error) => panic!("Ahhh no find of the file! {:?}", error),
    // };

}

fn read_username_from_file_1() -> Result<String, io::Error> {
    let username_file_result = File::open("this_file_might_not_exist.txt");
    let mut username_file = match username_file_result {
        Ok(file) => file,
        Err(e) => return Err(e),  // <-- early return
    };

    let mut username = String::new();
    match username_file.read_to_string(&mut username) {
        Ok(_) => Ok(username),
        Err(e) => Err(e),
    }
}

// Using the `?` suffix, this is much shorter
// `?` can be used wherever the function has a compatible return type
// (`Result` or `Option` , of a type that implements `FromResidual`)
fn read_username_from_file_2() -> Result<String, io::Error> {
    let mut username_file = File::open("this_file_might_not_exist.txt")?;
    let mut username = String::new();
    username_file.read_to_string(&mut username)?;
    Ok(username)
}

// Chaining calls, this is even shorter!
fn read_username_from_file_3() -> Result<String, io::Error> {
    let mut username = String::new();
    File::open("this_file_might_not_exist.txt")?.read_to_string(&mut username)?;
    Ok(username)
}

// For functions with an `Option` return, a `None` will be returned early
fn last_char_of_first_line(text: &str) -> Option<char> {
    text.lines().next()?.chars().last()
    // If we wanted this function to skip `\n` chars,
    // a match statement is more appropriate for more complex logic
}


// (In practice, `fs::read_to_string("filename.txt")` is easier)
