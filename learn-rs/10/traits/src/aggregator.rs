use std::fmt::{Display, Debug};

pub trait Summary {  // Declare a public trait called Summary
    // declare method signatures that describe the
    // behaviors of the types with this trait
    // fn summarize(&self) -> String;
    // Or... you can write a default implementation
    // fn summarize(&self) -> String {
    //     String::from("(Would you like to know more?..)")
    // }
    // .. A default implementation that calls a method
    fn summarize(&self) -> String {
        format!("(from {} Would you like to know more?..)", self.summarize_author())
    }
    fn summarize_author(&self) -> String;
}
// The compiler will enforce all types implementing this trait
// have defined a method `summarize`

pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    // fn summarize(&self) -> String {
    //     format!("{}, by {} ({})", self.headline, self.author, self.location)
    // }
    // To use all default implementation for traits what have all default,
    // you can simply leave this block empty
    // `impl Summary for NewsArticle {}`
    fn summarize_author(&self) -> String {
        format!("{}!", self.author)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {  // This overrides the default method
        format!("{}: {}", self.username, self.content)
    }
    // but we still need to  specify the required method, (it might be used by the coder)
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
}

// Traits can also be used as function parameters!
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize())
}
// This function will accept any item that implements the Summary trait
// and will reject anything that doesn't
//
// The syntax `&impl Summary` is shorthand for a "trait bound"
// pub fn notify<T: Summary>(item: &T) {
//     println!("Breaking news! {}", item.summarize())
// }
//
// Although this is more verbose, it's still useful with multiple parameters
// or more complex trait restrictions
pub fn notify2<T: Summary>(item1: T, item2: T) {  // Same type, different types should have impl SH
    println!("2 Breaking newses! {}, {}", item1.summarize(), item2.summarize())
}

// Multiple traits can be specified with +
pub fn pretty_notify(item: &(impl Summary + Display)) {
    println!("Breaking news! {}", item.summarize())
}

// Complex bounds are better expressed as `where` clauses
// And returned traits are implemented in the way you'd expect
// (But function must still only return one type!)
fn some_fucntion<T, U>(t: &T, u: &U) -> impl Summary
where
    T: Display + Clone,
    U: Clone + Debug,
{
    println!("a function!");
    NewsArticle {
        headline: String::from("foo"),
        location: String::from("foo"),
        author: String::from("foo"),
        content: String::from("foo"),
    }
}


struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}


// Conditional methods are only implemented if the inner type has the required traits
impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("The largest member is x = {}", self.x);
        } else {
            println!("The largest member is y = {}", self.y);
        }
    }
}
