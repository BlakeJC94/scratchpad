// Rust doesn't prevent creating traits having methods that
// share names with methods in other traits

// If shared method names are needed, the method source
// should be specified though

// For example, we have 2 traits with the method `fly`
trait Pilot {
    fn fly(&self);
}
trait Wizard {
    fn fly(&self);
}

struct Human;

// Lets implement these traits for the `Human` struct
impl Pilot for Human {
    fn fly(&self) {
        println!("This is your captain speaking.");
    }
}
impl Wizard for Human {
    fn fly(&self) {
        println!("ZOOT!");
    }
}

// Lets also implement a default method
impl Human {
    fn fly(&self) {
        println!("*Waving arms furiously*");
    }
}

fn main1() {
    let person = Human;
    person.fly();  // This calls (3)
    Pilot::fly(&person);  // This calls (1)
    Wizard::fly(&person);  // This calls (2)
    Human::fly(&person);  // This also calls (1)
}

// But what if the associated function doesn't have a `&self` parameter??

trait Animal {
    fn baby_name() -> String;
}

struct Dog;
impl Dog {
    fn baby_name() -> String {
        String::from("Spot")
    }
}
impl Animal for Dog {
    fn baby_name() -> String {
        String::from("puppy")
    }
}

fn main2() {
    println!("A baby dog is called a {}", Dog::baby_name());  // `Spot`
    // ERROR
    // println!("A baby dog is called a {}", Animal::baby_name());
    // FIX specify which method to use via <Type as Trait> syntax
    println!("A baby dog is called a {}", <Dog as Animal>::baby_name());  // `puppy`
}

fn main() {
    // main1();
    main2();
}

