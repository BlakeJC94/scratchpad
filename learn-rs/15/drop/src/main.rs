// We can also customise what happen when the resource is freed
// for any type using the `Drop` trait

// This would be useful for close connection and cleaning up files and shit

fn main() {
    main1();
    main2();
}

struct CustomSmartPointer {
    data: String,
}

impl Drop for CustomSmartPointer {
    fn drop(&mut self) {
        println!("Dropping CustomSmartPointer with data `{}`!", self.data);
    }
}

fn main1() {
    println!("MAIN1");
    let c = CustomSmartPointer {
        data: String::from("my stuff"),
    };
    let d = CustomSmartPointer {
        data: String::from("my other stuff"),
    };
    println!("CustomSmartPointers created");
    // Wonderful! The drop code gets called automatically in the reverse order of symbol assignment
}

// What if we need to manually override the automatic drop and free something from memory?
fn main2() {
    println!("MAIN2");
    let c = CustomSmartPointer {
        data: String::from("my stuff"),
    };
    println!("CustomSmartPointers created");
    // ERROR:
    // c.drop();
    // Looks like manual calls of this are verboten!
    // This is so we can avoid double-free issues
    // FIX:
    drop(c);
    // This calls c.drop() for us!
    // Conveniently, this is what's called when a symbol exits scope
    // So this would be a great way to manage resources!
    println!("End of function");
}

