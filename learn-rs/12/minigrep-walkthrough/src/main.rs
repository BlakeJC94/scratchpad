use std::env;
use std::process;

use minigrep::Config;

// fn main() {
//     let args: Vec<String> = env::args().collect();  // env::args_os is for complex unicode inputs

//     // println!("args = {:?}", args);
//     // dbg!(args);  // The dbg macro is very useful for doing some quick printing
//     // From the dprint, looks likes its the same spec as C:
//     // - 0 is for the command name
//     // - 1.. is all the args provided

//     // TODO is there a way to safely access these?
//     let query = &args[1];
//     let file_path = &args[2];
//     println!("Searching for '{}'", query);
//     println!("In filepath '{}'", file_path);

//     let contents = fs::read_to_string(file_path)
//         .expect("Should have been able to read the file.");

//     println!("With text:\n{}", contents)
// }
// Before we go further, this is a good point to start putting some of there responsibilities into
// smaller functions because:
// - main has too many responsibilities
// - query and filepath are config variables, whereas content is a result used in the core logic
// - expect isn't robust enough to print out specific reasons for why reading failed
// - index out of bounds errors are likely

// Best practice is to
// - split into main.rs and lib.rs
//     - lib.rs has the logic resources, main.rs should be the driver
//     - Provided the parsing logic is small, this can stay in main.rs
//

fn main() {
    let args: Vec<String> = env::args().collect();

    // let config = parse_config(&args);
    // let config = Config::build(&args).unwrap();
    let config = Config::build(&args).unwrap_or_else(|err| {
        // Yooo what this?? This looks like an anonymous func, will be covered in Ch13
        // println!("Problem parsing args : {}", err);
        process::exit(1);
    });

    // Lets also split out the processing login into a separate `run` function
    // And do errors better here too
    if let Err(e) = minigrep::run(config) {
        eprintln!("Application error: {}", e);  // It's wise to print errors to stderr!!
        process::exit(1);
    }
}


// struct Config {
//     query: String,
//     file_path: String,
// }

// // fn parse_config(args: &[String]) -> Config {
// //     let query = args[1].clone();
// //     let file_path = args[2].clone();
// //     // NOTE: cloning is an imperfect solution to ownership woes,
// //     // But a working program is better than a hyperoptimised one that doesn't work
// //     Config {query, file_path}
// // }
// // This is good, but why don't we use a constructor?

// impl Config {
//     // Use `build` instead of `new`, since `new` is not expected to fail
//     fn build(args: &[String]) -> Result<Config, &'static str> {
//         // Lets also catch some errors
//         if args.len() < 3 {
//             // panic!("Not enough arguments. Usage: `$ minigrep <query> <file_path>`")
//             // Panics return a lot of useless info, change the resuturn to a Result
//             return Err("Not enough arguments. Usage: `$ minigrep <query> <file_path>`")
//         }
//         let query = args[1].clone();
//         let file_path = args[2].clone();
//         // Config { query, file_path }
//         Ok(Config { query, file_path })
//     }
// }

// // This Box<..> shit is a "trait object", will be covered in Ch17
// fn run(config: Config) -> Result<(), Box<dyn Error>> {
//     println!("Searching for '{}'", config.query);
//     println!("In filepath '{}'", config.file_path);
//     let contents = fs::read_to_string(config.file_path)?;
//     // Instead of using expect, `?` will early return an error

//     println!("With text:\n{}", contents);
//     Ok(())
// }


// Now lets put all this into lib and simplify main.rs
