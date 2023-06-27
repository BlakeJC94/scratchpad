const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
// only values that can be computed at compile time can be declared as const
//
fn main() {
    // VARIABLES
    //
    // let x = 5;
    // println!("The value of x is: {x}");
    // x = 6;
    // println!("The value of x is: {x}");
    // Error: can't assign twice to to variable

    // let mut x = 5;
    // println!("The value of x is: {x}");
    // x = 6;
    // println!("The value of x is: {x}");
    // This is all G

    // SHADOWING
    let x = 5;
    let x = x + 1;
    {
        let x = x * 2;
        println!("The value of x in the inner scope is: {x}");  // 12
    }
    println!("The value of x is: {x}");  // 6
    // Using the let kword and the same name, simple transformas can be done on an immuntable
    // Because this is technically making a new immutable variable each time
    // TODO vim selecting 11z= instead of 1z=?
    //
    // This is handy in cases when the type needs to change
    // (mut variable can't have their type changed)
    // let mut spaces = "   ";
    // spaces = spaces.len();
    // ERROR
    let spaces = "   ";
    let spaces = spaces.len();
}

