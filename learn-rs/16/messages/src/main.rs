// A common pattern for safe concurrency is messaging
// "don't share memory between threads, share messages!"
use std::{sync::mpsc, thread, time::Duration};  // multiple producer, single consumer

// Channels are used to send data from one channel to another
fn main() {
    let (tx, rx) = mpsc::channel();
    // tx = transmitter (upstream), rx = reciever (downstream)
    // We can even close the transmitter and put it on another thread!
    let tx_copy = tx.clone();

    // ERROR: infinite loop
    // let recieved = rx.recv().unwrap();  // recv will block the main thread until
    // FIX: try_recv wont vblock the main process, just returns an error
    let recieved = rx.try_recv().unwrap_or(String::from("(nuthin)"));
    println!("Got message: {}", recieved);

    // Recall that `move` is required to satisify the closure ownership rules
    // Let's transfer a value from a thread to the main thread
    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();  // This will catch errors such as a dead thread
            thread::sleep(Duration::from_secs(1));
        }

        // ERROR:
        // println!("{:?}", vals);  // val was moved!
        // This stops us from doing antisocial things like modifying or dropping values
        // after they've been sent
    });

    // We can even close the transmitter and put it on another thread!
    thread::spawn(move || {
        let vals = vec![
            String::from("more"),
            String::from("messages"),
            String::from("from"),
            String::from("another"),
            String::from("thread"),
        ];

        for val in vals {
            tx_copy.send(val).unwrap();  // This will catch errors such as a dead thread
            thread::sleep(Duration::from_millis(500));
        }

        // ERROR:
        // println!("{:?}", vals);  // val was moved!
        // This stops us from doing antisocial things like modifying or dropping values
        // after they've been sent
    });


    for recieved in rx {
        println!("Got message: {}", recieved);
    }

}
