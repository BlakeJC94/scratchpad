// `HashMap<K, V>` maps values of type K to values of type V
// Check out the stdlib docs for all the deets
use std::collections::HashMap;


fn main() {
    // A way to create a hash map is by making a new one and inserting keys
    let mut scores = HashMap::new();
    let team_name = String::from("blue");
    let team_score = 10;
    scores.insert(team_name, team_score);
    scores.insert(String::from("yellow"), 50);
    println!("This is a hash map: {:?}", scores);

    // Like vectors, they store data on the heap
    // Keys must have the same type, all values must also have the same type

    // To access values, use the get method
    let other_team_name = String::from("yellow");
    let score = scores.get(&other_team_name);
    println!("score for {other_team_name} is {score:?}");
    println!("Whoops, it's a Some(..), should unwrap it safely");
    let score = score.copied().unwrap_or(0);  // `copied` converts Option<&i32> to Option<i32>
    println!("score for {other_team_name} is {score}");

    println!("Iterating is simple as well, but the order is arbitrary (changes each time?)");
    for (key, value) in &scores {
        println!("  {}:  {}", key, value);
    }

    // When I add keys and values,
    // ownership of heap data is transferred to the map
    // ERROR:
    // println!("{}", team_name)
    println!("{}, no error since it's on the stack", team_score);

    // values can be overwritten
    scores.insert(String::from("blue"), 20);
    println!("This is a hash map: {:?}", scores);

    // Or keys can be added only if it doesn't already exist in the map
    scores.entry(String::from("yellow")).or_insert(100);
    scores.entry(String::from("green")).or_insert(100);
    println!("This is a hash map: {:?}", scores);

    // Updating values of a hashmap
    let text = "foo bar baz wumbo zoop foo foo baz";
    let mut word_count = HashMap::new();
    for word in text.split_whitespace() {
        let count = word_count.entry(word).or_insert(0);
        *count += 1;  // Don't forget to deref references
    }
    println!("word count for {text},");
    println!("{word_count:?}");

}
