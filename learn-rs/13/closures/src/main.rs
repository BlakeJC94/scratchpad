// - Exclusive shirt giveaway
// - If user has spcified their faveourtite color, send that color
// - Otherwise, send the color thats highest in stock

fn main() {
    main1();
    main2();
    main3();
    main4();
    main5();
    main6();
}


use std::{thread, time::Duration};

#[derive(Debug, PartialEq, Copy, Clone)]
enum ShirtColor {
    Red,
    Blue,
}

struct Inventory {
    shirts: Vec<ShirtColor>,
}

impl Inventory {
    fn givewaway(&self, user_preference: Option<ShirtColor>) -> ShirtColor {
        // Here's a really basic closure
        // It takes no parameters, and it returns `self.most_stocked()` when evaluated
        // Rust automatically figures out the return type from `Option<ShirtColor>` in this case
        user_preference.unwrap_or_else(|| self.most_stocked())
        // What's interesting here is that the scope of this tiny function
        // includes the scope of the function itself -- we referred to `self` without issue!
    }

    fn most_stocked(&self) -> ShirtColor {
        let mut num_red = 0;
        let mut num_blue = 0;

        for color in &self.shirts {
            match color {
                ShirtColor::Red => num_red += 1,
                ShirtColor::Blue => num_blue += 1,
            }
        }
        if num_red > num_blue {
            ShirtColor::Red
        } else {
            ShirtColor::Blue
        }
    }
}

fn main1() {
    println!("MAIN1");
    let store = Inventory {
        shirts: vec![ShirtColor::Blue, ShirtColor::Red, ShirtColor::Blue],
    };

    let user_pref1 = Some(ShirtColor::Red);
    let giveaway = store.givewaway(user_pref1);
    println!(
        "The user with preference {:?} gets {:?}",
        user_pref1, giveaway
    );

    let user_pref2 = None;
    let giveaway = store.givewaway(user_pref2);
    println!(
        "The user with preference {:?} gets {:?}",
        user_pref1, giveaway
    );


}

// Illustration of the sliding scale between functions and closures
// fn  add_one_v1   (x: u32) -> u32 { x + 1 }
// let add_one_v2 = |x: u32| -> u32 { x + 1 };
// let add_one_v3 = |x|             { x + 1 };
// let add_one_v4 = |x|               x + 1  ;
// The types for the params will be inferred from the first time it's called
//
//
//
fn main2() {
    println!("MAIN2");
    let example_closure = |x| x;
    let s = example_closure(String::from("foo"));
    println!("input/return type of `example_closure` inferred as string: {s}");
    // ERROR:
    // let n = example_closure(5);

    let expensive_closure = |num: u32| -> u32 {
        println!("Calculating something slowly...");
        thread::sleep(Duration::from_secs(2));
        num
    };
    let n = expensive_closure(4);
    println!("n is {n}")
}

fn main3() {
    println!("MAIN3");
    let list = vec![1, 2, 3];
    println!("Before defining closure: {:?}", list);

    let only_borrows = || println!("From closure: {:?}", list);

    println!("Before calling closure: {:?}", list);
    only_borrows();
    println!("After calling closure: {:?}", list);
}

fn main4() {
    println!("MAIN4");
    let mut list = vec![1, 2, 3];
    println!("Before defining closure: {:?}", list);

    let mut borrows_mutably = || list.push(7);

    // ERROR: `list` is now inside closure, so can;t be used until the closure is finished
    // println!("Before calling closure: {:?}", list);
    borrows_mutably();
    println!("After calling closure: {:?}", list);
}

// We can force closures to take ownsership instead of borrowing the values used in the env,
// even though it's not strictly needed.
// This pattern is useful when passing a closure to a new thread to mode data so it's owned by the
// new thred
fn main5() {
    println!("MAIN5");
    let list = vec![1, 2, 3];
    println!("Before defining closure: {:?}", list);
    // New thread may finish before/after main thread
    // So transferring ownership like this protects against another class of invalid refs
    // (Error will happen without `move` kword)
    let proc = move || println!("Hello from thread : {:?}", list);
    // ERROR
    // println!("{list:?}");
    thread::spawn(proc)
        .join()
        .unwrap();
}


#[derive(Debug)]
struct Rect {
    width: u32,
    height: u32,
}

fn main6() {
    println!("MAIN6");
    let mut list = [
        Rect { width:10, height:3 },
        Rect { width:3, height:5 },
        Rect { width:7, height:12 },
    ];

    println!("Before sort: {:#?}", list);
    list.sort_by_key(|r| r.width);
    println!("After sort: {:#?}", list);
}

// Traits are auto-implemented on closures depending on the context. These traits are:
//
// FnOnce:
//     - applies to closures that can be called once.
//     - All closures implement at least this trait, because all closures can be called.
//     - A closure that moves captured values out of its body will only implement FnOnce and none of the other Fn traits, because it can only be called once.
// FnMut:
//     - applies to closures that don’t move captured values out of their body
//         - (but that might mutate the captured values).
//     - These closures can be called more than once.
// Fn:
//     - applies to closures that don’t move captured values out of their body
//       and that don’t mutate captured values,
//         - as well as closures that capture nothing from their environment.
//     - These closures can be called more than once without mutating their environment, which is important in cases such as calling a closure multiple times concurrently.

