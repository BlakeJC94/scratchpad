use std::{collections::HashMap, env::args};

fn main() {
    let integer_list = parse_input();

    let median = get_median(integer_list.clone());
    println!("median = {}", median);

    let mode = get_mode(integer_list.clone());
    println!("mode = {}", mode);
}

fn parse_input() -> Vec<i32> {
    let mut integer_list: Vec<i32> = Vec::new();
    for (i, arg) in args().enumerate() {
        if i == 0 {
            continue;
        }
        let integer: i32 = arg.parse().expect("Not a number!");
        integer_list.push(integer);
    }
    integer_list
}

fn get_median(mut list: Vec<i32>) -> i32 {
    if !list.is_empty() {
        return 0
    }
    list.sort();
    let median_idx = list.len() / 2;
    list[median_idx]
}

fn get_mode(list: Vec<i32>) -> i32 {
    let mut mode = 0;

    let mut value_count: HashMap<i32, i32> = HashMap::new();
    for v in &list {
        let count = value_count.entry(*v).or_insert(0);
        *count += 1; // Don't forget to deref references
    }
    let mut max_count = -1;
    for (k, v) in &value_count {
        if max_count < *v {
            max_count = *v;
            mode = *k;
        }
    }

    mode
}
