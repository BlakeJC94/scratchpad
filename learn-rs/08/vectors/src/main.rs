// Vectors are useful for when you want a growable array on the heap

fn main() {
    println!("Vectors!");

    let mut v1: Vec<i32> = Vec::new();  // Type annot needed since we're making an empty vector
    let v2 = vec![1, 2, 3];  // Or you could use the `vec!` macro on an array
    let mut v3 = vec![1, 2, 3];  // Or you could use the `vec!` macro on an array
    // `v2` didn't need type annot since the values provided already had an i32 type
    println!("It's a vector! v1={:?}", v1);
    println!("It's a vector! v2={:?}", v2);  // <- `{:?}` is needed

    // Here's how to iterate over immutable references to values
    for x in &v3 {
        println!("v3: Vector value = {x:?}");
    }

    // Here's how to iterate over mut references to values and change values
    for x in &mut v3 {
        *x *= 10;  // <- de-referencing is discussed more in ch15
        println!("v3: Vector value (after change) = {x:?}");
    }

    // To update a vector, make sure it's mutable first! Then just use push
    v1.push(5);
    v1.push(6);
    v1.push(7);
    for x in v1.iter() {
        println!("v1: Vector value after pushing = {x:?}");
    }

    // To index into a vector, (assuming it exists)
    let idx = 2;
    let elem: &i32 = &v1[idx];
    println!("Element {idx} of v1 is {elem}");
    // This will panic fail if v1 doesn't have a value at this index

    // Recall that a reference can't have its data modified when it's in scope
    // This is to prevent simetanous modification and usage, a common source of bugs!
    // (This also occurs when trying to add/remove vectors in a for-loop)
    // ERROR:
    // let elem: &i32 = &v1[idx];
    // v1.push(5);
    // println!("Element {idx} of v1 is {elem}");


    // To "get" from a vector, (without assuming it exists)
    let elem_3: Option<&i32> = v1.get(3);
    println!("Does v1 have an element in index 3?");
    match elem_3 {
        Some(elem) => println!("Dayum we got en element here! {elem}"),
        None => println!("Nah mate, v1 is too short"),
    }

    // Like arrays, vectors must have a single type
    // To store multiple types, an enum could be used
    #[derive(Debug)]
    enum SheetCell {
        Int(i32),
        Float(f64),
        Text(String)
    }
    let row = vec![
        SheetCell::Int(3),
        SheetCell::Text(String::from("blue")),
        SheetCell::Float(10.12),
    ];
    println!("row = {row:?}");

}  // Like all heap objects, once v1, v2, v3 are out of scope, they're freed from mem


// Also FR, the collections documetation is so nice!
