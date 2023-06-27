fn main() {
    main1();
    main2();

}

fn main1() {
    // Say we want to find the largest number in a list
    let number_list = vec![34, 50, 25, 100, 65];
    let mut largest = &number_list[0];
    for number in &number_list {
        if number > largest {
            largest = number;
        }
    }
    println!("The largest number is {}", largest);

    // Say now we need to find the largest number across 2 lists
    let number_list = vec![102, 34, 6000, 89, 54, 2, 43, 8];
    let mut largest = &number_list[0];
    for number in &number_list {
        if number > largest {
            largest = number;
        }
    }
    println!("The largest number is {}", largest);
    // This works, but gross this uses a lot of duplicate code..
    // Let's define a function that operates on ay list of integers passed in as a parameter
}

fn largest(list: &[i32]) -> &i32{
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

fn main2() {
    let number_list = vec![34, 50, 25, 100, 65];

    let result = largest(&number_list);
    println!("The largest number is {}", result);

    let number_list = vec![102, 34, 6000, 89, 54, 2, 43, 8];
    let result = largest(&number_list);
    println!("The largest number is {}", result);
}

// This is the same process we'll use to write generics to reduce code duplication
// Like how functions allow code to act on concrete types,
// Generics allow code to act on abstract types
// E.g. how would we use a single piece of code to find the largest i32 and the largest char byte?
