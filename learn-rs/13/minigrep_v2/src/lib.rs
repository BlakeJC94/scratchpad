use std::error::Error;
use std::{env, fs};

pub struct Config {
    pub query: String,
    pub file_path: String,
    pub ignore_case: bool,
}

impl Config {
    // Use `build` instead of `new`, since `new` is not expected to fail
    // pub fn build(args: &[String]) -> Result<Config, &'static str> {
    //     if args.len() < 3 {
    //         return Err("Not enough arguments. Usage: `$ minigrep <query> <file_path>`");
    //     }
    //     let query = args[1].clone();
    //     let file_path = args[2].clone();

    //     // Bonus: override env var with a flag!
    //     let ignore_case = if args.len() == 4 {
    //         args[3].contains("-i")
    //     } else {
    //         match env::var("MINIGREP_IGNORE_CASE") {
    //             Ok(val) => val == "1" || val.to_lowercase().starts_with('t'),
    //             Err(_) => false,
    //         }
    //     };

    //     Ok(Config {
    //         query,
    //         file_path,
    //         ignore_case,
    //     })
    // }

    // Need to update the input type to accept an iterator
    pub fn build(mut args: impl Iterator<Item = String>) -> Result<Config, &'static str> {
        args.next(); // Skip arg 0

        let query = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get a query"),
        };

        let file_path = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get a file path"),
        };

        // Bonus: override env var with a flag!
        let ignore_case = match args.next() {
            Some(arg) => arg.contains("-i"),
            None => match env::var("MINIGREP_IGNORE_CASE") {
                Ok(val) => val == "1" || val.to_lowercase().starts_with('t'),
                Err(_) => false,
            },
        };

        Ok(Config {
            query,
            file_path,
            ignore_case,
        })
    }
}

// This Box<..> shit is a "trait object", will be covered in Ch17
pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(config.file_path)?;
    // Instead of using expect, `?` will early return an error

    let results = if config.ignore_case {
        search_case_insensitive(&config.query, &contents)
    } else {
        search(&config.query, &contents)
    };

    for line in results {
        println!("{line}");
    }
    Ok(())
}

// Note that we should use a lifetime here to indicate that
// theref to the result depends on the ref to the contents, not query
// (We can write this more concisely with closures!)
// pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
//     let mut results = Vec::new();
//     for line in contents.lines() {
//         if line.contains(query) {
//             results.push(line);
//         }
//     }
//     results
// }

// (We can write this more concisely with closures!)
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    contents.lines().filter(|l| l.contains(query)).collect()
}

pub fn _search<'a>(query: &str, contents: &'a str, case_insensitive: bool) -> Vec<&'a str> {
    contents
        .lines()
        .filter(|l| {
            if case_insensitive {
                l.to_lowercase().contains(&query.to_lowercase())
            } else {
                l.contains(query)
            }
        })
        .collect()
}

pub fn search_case_insensitive<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    contents
        .lines()
        .filter(|l| l.to_lowercase().contains(&query.to_lowercase()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_sensitive() {
        let query = "duct";
        let contents = "\
Rust:
safe, fast, productive.
Pick three.
Duct tape.";

        assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    }

    #[test]
    fn case_insensitive() {
        let query = "rUsT";
        let contents = "\
Rust:
safe, fast, productive.
Pick three.
Trust me.";

        assert_eq!(
            vec!["Rust:", "Trust me."],
            search_case_insensitive(query, contents)
        );
    }
}
