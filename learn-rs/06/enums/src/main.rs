fn main() {
    main1();
    main2();
    main3();
    main4();
}
// Struct are useful for grouping related data together
// But enums are useful for defining the possibilities of fields
#[derive(Debug)]  // <-- I put this in to make a print statement work
enum IpAddrKind {
    V4,
    V6,
}

// Now we can define functions that act on IP Addresses
fn route(ip_kind: IpAddrKind) {
    println!("it's a {:?} address!", ip_kind);
}
// This function will work accept any variant of IpAddrKind

fn main1() {
    println!("MAIN1");
    let four = IpAddrKind::V4;
    route(four);
    let six = IpAddrKind::V6;
    route(six);
}

// Maybe we can use this enum in a struct to couple together data and type!
#[derive(Debug)]  // <-- I put this in to make a print statement work
struct IpAddrStruct {
    kind: IpAddrKind,
    address: String,
}

fn main2() {
    println!("MAIN2");
    let home = IpAddrStruct {
        kind: IpAddrKind::V4,
        address: String::from("127.0.0.1"),
    };
    println!("home is where the {:?} is!", home);
}

// But we can do better!
// We can put data directly into an enum
#[derive(Debug)]  // <-- I put this in to make a print statement work
enum IpAddrEnum {
    V4(u8, u8, u8, u8),  // V4 addresses always have 4 numbers between [0, 255]
    V6(String),
}
// If we wanted to, we could even put structs in here!!

fn main3() {
    println!("MAIN3");
    let loopback = IpAddrEnum::V6(String::from("::1"));
    println!("loopback is {:?}", loopback);
    let home = IpAddrEnum::V4(127, 0, 0, 1);
    println!("home is where the {:?} is!", home);
}

// Lets look at another example
// 4 types, Quit has no data
#[derive(Debug)]
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
//equiv to
// struct QuitMessage; // unit struct
// struct MoveMessage {
//     x: i32,
//     y: i32,
// }
// struct WriteMessage(String); // tuple struct
// struct ChangeColorMessage(i32, i32, i32); // tuple struct

// What about methods? Well we can do that too!
impl Message {
    fn call(&self) {
        println!("This message is {:?}", &self);
    }
}
fn main4() {
    println!("MAIN4");
    let message = Message::Quit;
    message.call();
    let message2 = Message::Move{x:32, y:34};
    message2.call();
}
