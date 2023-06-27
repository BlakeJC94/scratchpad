// This time, lets encode state and behaviour as types
// As an alternate implemntation to the OOP appraoch
// that utilises more of the strengths of Rust

use bloop2::Post;

fn main() {
    let mut post = Post::new();
    post.add_text("I ate a piece of toast for lunch today");

    let post = post.request_review();
    let post = post.approve();
    assert_eq!("I ate a piece of toast for lunch today", post.content());
}
