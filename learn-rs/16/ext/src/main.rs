// There's very fw concurrency features in Rust besides the stdlib
// But there are traits `Sync` and `Send` that allow extensions to concurrency

// Send is used to show Rust how to send data between threads
// (not implemented for RC, for example, because it's not thread-safe)

// Sync allows access from multiple threads
// (also not implemented for RC, for example, because it's not thread-safe)

// Structs that are composed entirely of objects with Sync/Send
// automatcially have sync/send implemented.
// Manual intervention is pretty rare (and a bit unsafe)


fn main() {
    println!("Hello, world!");
}
