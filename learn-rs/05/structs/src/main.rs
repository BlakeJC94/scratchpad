// Structs are similar to tuples
// - Basic collection
// - Non-uniform types
// Struct components are named instead of positional
// And can be indexed with dot notation
// (Recall tuples are indexed the same way)

struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

// Structs can also be delcared from tuples
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

// If you don't have any data to store, but still want ot record some tympe of traint,
// use a unit type struct
struct AlwaysEqual;


fn main() {
    let mut user1 = User {
        email: String::from("foo@barbaz.com"),
        username: String::from("FooBar"),
        active: true,
        sign_in_count: 3,
    };
    println!("User {} has signed in {} times", user1.username, user1.sign_in_count);
    user1.sign_in_count += 1;
    println!("User {} has signed in {} times", user1.username, user1.sign_in_count);
    //
    // Only entire instances of structs can be mutable, not specific fields
    //
    // Lets use a constructure to make instances of a struct
    let user2 = create_user(String::from("wumbo@floob.com"), String::from("Flooba"));
    println!("User {} has signed in {} times", user2.username, user2.sign_in_count);

    // We can also create an instance from another instance
    // by prefixing the last instance with .. at the end (no comma)
    let user3 = User {
        email: String::from("another@example.com"),
        ..user1
    };
    println!("User {} has email {} instead of {}", user3.username, user3.email, user1.email);
    // Note that usage of = tends to move the data
    // So now user1 can't be used because we moved heap data
    // ERROR:
    // println!("User {} has email {}", user1.username, user1.email);
    let black = Color(0,0,0);
    let origin = Point(0,0,0);
}

fn create_user(email: String, username: String) -> User {
    User {
        // email: email,
        email,
        // username: username,
        username,  // Since these variables have the names as the fields, shorthand can be used!
        active: true,
        sign_in_count: 1,
    }
}
