fn main() {
    main4();
}

fn main1() {
    println!("MAIN 1");
    let width1 = 30;
    let height1 = 50;

    println!("The area of the rect is {}", area1(width1, height1));
}

fn area1(width: u32, height: u32) -> u32 {
    width * height
}

// area1 takes in two distinct arrgs, one would be better
// ---- Refactor with tuples

fn main2() {
    println!("MAIN 2");
    let rect = (30, 50);

    println!("The area of the rect is {}", area2(rect));
}

fn area2(rect: (u32, u32)) -> u32 {
    rect.0 * rect.1
}

// area2 is stringly coupled with the order of the args
// Lets add meaning to the fields with a struct
// ---- Refactor with structs
struct Rect {  // Structs should be at the tops of the file (style nit)
    width: u32,
    height: u32,
}
fn main3() {
    println!("MAIN 3");
    let rect = Rect {
        width: 30,
        height: 50,
    };

    println!("The area of the rect is {}", area3(&rect));
}

fn area3(rect: &Rect) -> u32 {
    rect.width * rect.height
}

// This is better, but do we even need the area function?
// We can derive the Debug trait to the struct to print it with {:?} placeholder
// ---- Debugging with traits
#[derive(Debug)]  // #[..] adds an "attribute" to the code
struct RectWithDebug {
    width: u32,
    height: u32,
}

fn main4() {
    println!("MAIN 4");
    let scale = 2;
    let rect = RectWithDebug {
        width: dbg!(30 * scale),  // <- Useful macro
        height: 50,
    };
    // dbg! returns ownership

    // To print to stderr,
    dbg!(&rect);

    // To print to stdout, add the Debug trait and use this placeholder:
    println!("The rect is {:?}", rect);
    // {:?} is for a single line, {:#?} is for multiple lines
    // println!("The area of the rect is {:?}", rect);

}

fn area4(rect: &Rect) -> u32 {
    rect.width * rect.height
}
// It's be useful to put this area code as a method
