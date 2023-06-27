fn main() {
    valid_places_patterns_can_be_used();
}

// In some places, "Refutable" pattterns are allowed.
// Refutable ptterns may fail to match
// And some places require that patterns match

fn valid_places_patterns_can_be_used() {
    let foo = Some(3);
    println!("foo = {:?}", foo);

    println!("== Match arms ==");
    let bar = match foo {  // Match named variables
        Some(foo) if foo % 2 == 0 => Some(foo+10),
        Some(foo) => Some(foo+1),
        _ => None,  // Remember that _ matches anything
    }; // Must be exhaustive, cover all possibilities
    println!("foo = {:?}", foo);
    println!("bar = {:?}", bar);
    let x = 8;
    match x {  // Match literal values
        1 | 2 => println!("one or two"),  // Multiple patterns separated by |
        3 => println!("three"),
        4..=8 => println!("between four and eight"),
        // 'a'..='j' => println!("between 'a' and 'j'"),
        _ => println!("anything"),
    }

    enum Message {
        Hello { id: i32 },
    }

    let msg = Message::Hello { id: 5 };

    match msg {
        Message::Hello {
            id: id_variable @ 3..=7,
        } => println!("Found an id in range: {}", id_variable),
        Message::Hello { id: 10..=12 } => {
            println!("Found an id in another range")
        }
        Message::Hello { id } => println!("Found some other id: {}", id),
    }

    let favorite_color: Option<&str> = None;
    let is_tuesday = false;
    let age: Result<u8, _> = "34".parse();

    println!("== If let expressions ==");
    if let Some(color) = favorite_color {
        println!("Using your favorite color, {color}, as the background");
    } else if is_tuesday {
        println!("Tuesday is green day!");
    } else if let Ok(age) = age {
        if age > 30 {
            println!("Using purple as the background color");
        } else {
            println!("Using orange as the background color");
        }
    } else {
        println!("Using blue as the background color");
    }

    println!("== While let loops ==");
    let mut stack = Vec::new();
    stack.push(1);
    stack.push(2);
    stack.push(3);

    while let Some(top) = stack.pop() {
        println!("{}", top);
    }

    println!("== For loops (IRREFUTABLE) ==");
    let v = vec!['a', 'b', 'c'];

    for (index, value) in v.iter().enumerate() {
        println!("{} is at index {}", value, index);
    }

    println!("== Let expressions (IRREFUTABLE) ==");
    let (x, y, z) = (10, 20, 30);
    println!("{}, {}, {}", x, y, z);
    // ERROR
    // let (a, b) = (4, 5, 6);
    // FIX
    let (a, b, ..) = (4, 5, 6);  // Also works in de-structuring structs

    println!("== Function parameters (IRREFUTABLE) ==");
    let point = (3, 5);
    print_coords(&point);


}

fn print_coords(&(x,y): &(i32, i32)) {
    println!("coords: {x}, {y}");
}
