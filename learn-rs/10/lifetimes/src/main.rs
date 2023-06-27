// Lifetimes are a type of generic used to control how ling a reference is valid for.
// Most of the time, this is simply inferred as [between definition, end of scope]

// (This is a concept not found in most other langs,
// so this might take some getting used to)

use std::fmt::Display;

// The main aim of lifetimes is to prevent dangling references
fn main() {
    main1();
    main2();
    main3();
    main4();
    main5();
}

// Lets annotate how the borrow checker works
fn main1() {
    println!("MAIN1");
    // ERROR: references data was dropped, lifetime of x is shorter than lifetime of r
    // let r;                  // --------+--'a
    // {                       //         |
    //     let x = 5;          // ---+-'b |
    //     r = &x;             //    |    |
    // }                       // ---+    |
    // println!("r: {}", r);   // --------+
    // FIX: adjust defs so lifetime of x is longer than lifetime of r
    let x = 5;            // ----------+-- 'b
    let r = &x;           // --+-- 'a  |
    println!("r: {}", r); // --+       |
}                         // ----------+

// Lets calculate the longest of two strings from their references
// We run into a surprising problem
fn main2() {
    println!("MAIN2");
    let string1 = String::from("abcd");
    let string2 = "xyz";
    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);
}

// ERROR: lifetime needed
// fn longest(x: &str, y: &str) -> &str {
//     if x.len() > y.len() {
//         x
//     } else {
//         y
//     }
// }
// FIX:
fn longest<'a>(x: &'a str, y: &'a str) -> &str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Why was that needed??
// Because in the original version, it's undetermined whether the
// return reference should be x or y.
// So the borrow checker cannot check the scopes of the inputs
// And thus cannot determine if the resulting references is always valid

// But if we specify these lifetimes, then it's clear that the input references
// have the same lifetimes, then the compiler is happy with that

// Lifetimes aren't much use in isolation,
// their main purpose is to relate the lifetimes of multiple inputs/outputs

// In particular, this prevents this situation from happening:
fn main3() {
    println!("MAIN3");
    let string1 = String::from("long string is long");
    // ERROR:
    // let result;
    // {
    //     let string2 = String::from("xyz");
    //     result = longest(string1.as_str(), string2.as_str());
    // }
    // println!("The longest string is {}", result);
    // FIX:
    {
        let string2 = String::from("xyz");
        let result = longest(string1.as_str(), string2.as_str());
        println!("The longest string is '{}'", result);
    }
}

// Note that the compiler will reject functions with unnecessary lifetime annotations

// Lifetimes can also be useful in structs
struct ImportantExcerpt<'a> {
    part: &'a str,
}
// Instances of `ImportantExcerpt` shouldn't outlive the references of the data they point to

// Syntax for lifetimes in methods are pretty simple too
// Question: Why is the liftime needed after the `impl` as well??
impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention please: {}", announcement);
        self.part
    }
}

fn main4() {
    println!("MAIN4");
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    let i = ImportantExcerpt {
        part: first_sentence,
    };
    let part = i.announce_and_return_part("foo");
    println!("part = '{}'", part)
}

// If you absolutely need a lifetime that lasts as long as the entire program,
// The special lifetime `'static` can be used.
// (But it's usually worth solving underlying issues before resorting to this)


// To finish, lets bring traits together with generics and lifetimes!
fn longest_with_annoncement<'a, T>(
    x: &'a str,
    y: &'a str,
    ann: T,
) -> &'a str
where
    T: Display,
{
    println!("announcement: '{}'", ann);
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main5() {
    println!("MAIN5");
    let string1 = String::from("abcd");
    let string2 = "xyz";
    let result = longest_with_annoncement(string1.as_str(), string2, 3);
    println!("The longest string is {}", result);
}
