use std::error::Error;
use std::{env, fs};

pub struct Config {
    pub query: String,
    pub file_path: String,
    pub ignore_case: bool,
}

impl Config {
    // Use `build` instead of `new`, since `new` is not expected to fail
    pub fn build(args: &[String]) -> Result<Config, &'static str> {
        // Lets also catch some errors
        if args.len() < 3 {
            // panic!("Not enough arguments. Usage: `$ minigrep <query> <file_path>`")
            // Panics return a lot of useless info, change the resuturn to a Result
            return Err("Not enough arguments. Usage: `$ minigrep <query> <file_path>`");
        }
        let query = args[1].clone();
        let file_path = args[2].clone();

        // let ignore_case = env::var("MINIGREP_IGNORE_CASE").is_ok();
        // Bonus: override env var with a flag!
        let ignore_case = if args.len() == 4 {
            args[3].contains("-i")
        } else {
            // env::var("MINIGREP_IGNORE_CASE").is_ok()
            match env::var("MINIGREP_IGNORE_CASE") {
                Ok(val) => val == "1" || val.to_lowercase().starts_with('t'),
                Err(_) => false,
            }
        };

        // Config { query, file_path }
        // Ok(Config { query, file_path })
        Ok(Config {
            query,
            file_path,
            ignore_case,
        })
    }
}

// This Box<..> shit is a "trait object", will be covered in Ch17
pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    // println!("Searching for '{}'", config.query);
    // println!("In filepath '{}'", config.file_path);
    let contents = fs::read_to_string(config.file_path)?;
    // Instead of using expect, `?` will early return an error

    // Adding some branching depending on the value of a flag
    let results = if config.ignore_case {
        search_case_insensitive(&config.query, &contents)
    } else {
        search(&config.query, &contents)
    };

    // println!("With text:\n{}", contents);
    for line in results {
        println!("{line}");
    }
    Ok(())
}

// Note that we should use a lifetime here to indicate that
// theref to the result depends on the ref to the contents, not query
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    let mut results = Vec::new();
    for line in contents.lines() {
        // println!("{line}")
        if line.contains(query) {
            results.push(line);
        }
    }
    results
}

pub fn search_case_insensitive<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    let mut results = Vec::new();
    for line in contents.lines() {
        // println!("{line}")
        if line.to_lowercase().contains(&query.to_lowercase()) {
            results.push(line);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn one_result() {
    //     let query = "duct";
    //     let contents = "\
    // Rust:
    // safe, fast, productive.
    // Pick three.";

    //     assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    // }

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
