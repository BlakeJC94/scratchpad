fn main() {
    main1();
}

fn main1() {
    println!("MAIN1");
    let v1 = vec![1, 2, 3];
    let v1_iter = v1.iter();
    // Iterators are lazy, nothing useful has happened by this point

    for val in v1_iter { // yay, no indexing required
        println!("Got: {}", val);
    }
    // ERROR: but they can only be used once!
    // for val in v1_iter { // yay, no indexing required
    //     println!("Again: {}", val);
    // }
    // Fix:
    let v1_iter = v1.iter();
    for val in v1_iter { // yay, no indexing required
        println!("Again: {}", val);
    }

    // All iterators implement the `next` method
    // We can call the `next` method on iterators (with mut)
    let mut v1_iter = v1.iter();
    println!("next item = {:?}, expected = {:?}", v1_iter.next(), Some(&1));
    println!("next item = {:?}, expected = {:?}", v1_iter.next(), Some(&2));
    println!("next item = {:?}, expected = {:?}", v1_iter.next(), Some(&3));
    println!("next item = {:?}, expected = {:?}", v1_iter.next(), None::<i32>);
    // `mut` wasn't needed for the for loop because
    // the loop takes ownership of the iterator
    // Calling the `next` changes internal state of the iterator

    // Use `into_iter` instead of `iter` if you want the values instead of refs
    // Use `iter_mut` instead of `iter` if you want mutable references

    // You can use closures to changes the iterators as well
    let v1_iter_map = v1.iter().map(|x| x + 1); // This does nothing until called
    for i in v1_iter_map {
        println!("Added 1 : {}", i);
    }

    // Or you could collect the items into a vec straight away
    let v2: Vec<_> = v1.iter().map(|x| x + 1).map(|x| x*x).collect();  // Note the type needed
    println!("v2 = {v2:?}")
}

#[derive(PartialEq, Debug)]
struct Shoe {
    size: u32,
    style: String,
}

// We can use filter with a bool-returning closure to filter an iterator
fn shoes_in_size(shoes: Vec<Shoe>, requested_shoe_size: u32) -> Vec<Shoe> {
    shoes.into_iter().filter(|s| s.size == requested_shoe_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_by_size() {
        let shoes = vec![
            Shoe {
                size: 10,
                style: String::from("foo"),
            },
            Shoe {
                size: 12,
                style: String::from("bar"),
            },
            Shoe {
                size: 13,
                style: String::from("baz"),
            },
        ];

        let in_my_size = shoes_in_size(shoes, 13);
        assert_eq!(
            in_my_size,
            vec![
                Shoe {
                    size: 13,
                    style: String::from("baz"),
                },
            ]
        );
    }
}



// fn main2() {
//     println!("MAIN2");
// }

