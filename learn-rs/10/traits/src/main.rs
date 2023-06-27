// Lets say we want to create an aggregator that can display
// summaries of news articles and tweets

// Lets put all the trait details in aggregator.rs

mod aggregator;

use crate::aggregator::{NewsArticle, Summary, Tweet};

// We can now implement the `Summary` trait on other new types we define here
// The main restriction is that you can't implement foreign traits onto local types
// implemented traits must always come from that same local space as the local type in question

fn main() {
    let tweet = Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("of course, as you probably already know, people"),
        reply: false,
        retweet: false,
    };

    println!("1 new tweet: {}", tweet.summarize());

    let article = NewsArticle {
        headline: String::from("Penguins win the Stanley Cup Championship!"),
        location: String::from("Pittsburgh, PA, USA"),
        author: String::from("Iceburgh"),
        content: String::from(
            "The Pittsburgh Penguins once again are the best \
             hockey team in the NHL.",
        ),
    };

    println!("New article available! {}", article.summarize());
}
