fn main() {
    main1();
    // main2();
    main3();
}

fn largest_i32(list: &[i32]) -> &i32 {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

fn largest_char(list: &[char]) -> &char {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

fn main1() {
    println!("MAIN1");
    let number_list = vec![34, 50, 25, 100, 65];

    let result = largest_i32(&number_list);
    println!("The largest number is {}", result);

    let char_list = vec!['y', 'm', 'a', 'q'];
    let result = largest_char(&char_list);
    println!("The largest char is {}", result);
}

// So finding the largest i32 and the largest char have pretty much exactly the same code
// Parameterised types can be done easily:
// fn largest<T>(list: &[T]) -> &T {
//     let mut largest = &list[0];
//     for item in list {
//         // ERROR: code wont compile until we restrict T with traits
//         if item > largest {
//             largest = item;
//         }
//     }
//     largest
// }

// fn main2() {
//     println!("MAIN2");
//     let number_list = vec![34, 50, 25, 100, 65];

//     let result = largest(&number_list);
//     println!("The largest number is {}", result);

//     let char_list = vec!['y', 'm', 'a', 'q'];
//     let result = largest(&char_list);
//     println!("The largest char is {}", result);
// }

// Where else can we use generic types?
// Can be used with structs, methods, enums
struct PointWithOnlyOneType<T> {
    x: T,
    y: T,
} // `PointWithOnlyOneType { x: 5, y: 10.0 }` will fail

#[derive(Debug)]
struct PointWithTwoTypesPossible<T, U> {
    x: T,
    y: U,
}

impl<T, U> PointWithTwoTypesPossible<T, U> {
    fn x(&self) -> &T {
        &self.x
    }
}

impl PointWithTwoTypesPossible<f32, f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

// Generic types can have more useful names (convention is capitalised though)
// Very useful when definining methodd on generic input/output types
impl<X1, Y1> PointWithTwoTypesPossible<X1, Y1> {
    fn mixup<X2, Y2>(
        self,
        other: PointWithTwoTypesPossible<X2, Y2>,
    ) -> PointWithTwoTypesPossible<X1, Y2> {
        PointWithTwoTypesPossible {
            x: self.x,
            y: other.y,
        }
    }
}

enum OptionIllustration<T> {
    Some(T),
    None,
}

enum ResultIllustration<T, E> {
    Ok(T),
    Err(E),
}

fn main3() {
    println!("MAIN3");
    let both_integer = PointWithOnlyOneType { x: 5, y: 10 };
    let both_float = PointWithOnlyOneType { x: 1.0, y: 4.0 };
    // ERROR
    // let integer_and_float = PointWithOnlyOneType { x: 5, y: 4.0 };
    // FIX
    let integer_and_float = PointWithTwoTypesPossible { x: 5, y: 4.0 };
    println!("p.x = {}", integer_and_float.x());
    let p = PointWithTwoTypesPossible { x: 5.0, y: 10.0 };
    println!("dist = {}", p.distance_from_origin());

    let p2 = PointWithTwoTypesPossible { x: 0.5, y: -1 };
    println!("p = {:?}", p);
    println!("p2 = {:?}", p2);
    println!("p2.mixup(p) = {:?}", p2.mixup(p));

}

// If you're writing duplicate code that only differs in type,
// consider using generic types
// The compiler will create all the required cases for you
