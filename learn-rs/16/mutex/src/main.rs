use std::{sync::{Mutex, Arc}, thread, time::Duration};

// Even though messages aare cool, what if we
// reeeeeaallly wanted to share memory between threads?
// Well it is possible, but Rust will make sure it's safe!
fn main() {
    // main1();
    // main2();
    main3();
    // main4();
}

// Let's start with a mutex in a single threaded context
fn main1(){
    println!("MAIN1");
    let m = Mutex::new(5);

    {
        // Thread needs to signal that it wants access
        // by attempting to acquire the lock
        // (This is how the mutex keeps track of
        // who has data access)
        let mut num = m.lock().unwrap();
        // Calling lock() will fail if another thread holding the lock panics
        *num = 6
    }
    // Once the value is out of scope,
    // the Mutex smart pointer will automatically unlock the data
    // when dropped

    println!("m = {:?}", m);
}

// fn main2(){
//     println!("MAIN2");
//     let counter = Mutex::new(0);
//     let mut handles = vec![];

//     // ERROR: This needs multiple ownership
//     for _ in 0..10 {
//         let handle = thread::spawn(move || {
//             let mut num = counter.lock().unwrap();
//             *num += 1;
//         });
//         handles.push(handle);
//     }

//     for handle in handles {
//         handle.join().unwrap();
//     }

//     println!("Result: {}", *counter.lock().unwrap())
// }

//Instead of using Rc, we need to use Arc.
// Because Rc is not thread-safe. Arc is, but it has a small performance penalty
fn main3(){
    println!("MAIN3");
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap())
}

// The risk that comes with Mutex is deadlocks,
// which occurs when two threads are waiting foe the other to drop the lock indefinitely
// ERROR:
fn main4(){
    println!("MAIN4");
    let counter1 = Arc::new(Mutex::new(0));
    let counter2 = Arc::new(Mutex::new(10));
    let mut handles = vec![];

    {
        let counter1 = Arc::clone(&counter1);
        let counter2 = Arc::clone(&counter2);
        let handle = thread::spawn(move || {
            let mut num = counter1.lock().unwrap();
            *num += 1;
            println!("Thread 1 holds counter1 lock and waits for counter2 lock");
            thread::sleep(Duration::from_millis(1000));
            let mut num = counter2.lock().unwrap();
            *num += 1;
            println!("Thread 1");
        });
        handles.push(handle);
    }

    {
        let counter1 = Arc::clone(&counter1);
        let counter2 = Arc::clone(&counter2);
        let handle = thread::spawn(move || {
            let mut num = counter2.lock().unwrap();
            *num += 1;
            println!("Thread 2 holds counter2 lock and waits for counter1 lock");
            thread::sleep(Duration::from_millis(1000));
            let mut num = counter1.lock().unwrap();
            *num += 1;
            println!("Thread 2");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter1.lock().unwrap());
    println!("Result: {}", *counter2.lock().unwrap());
}

// This looks pretty similar to data with multiple references (~RefCell)

// There's very fw concurrency features in Rust besides the stdlib
// But there are traits `Sync` and `Send` that allow extensions to concurrency

// Send is used to show Rust how to send data between threads
// (not implemented for RC, for example, because it's not thread-safe)

// Sync allows access from multiple threads
// (also not implemented for RC, for example, because it's not thread-safe)

// Structs that are composed entirely of objects with Sync/Send
// automatcially have sync/send implemented.
// Manual intervention is pretty rare (and a bit unsafe)
