use std::thread;
use std::time::Duration;

fn main() {
    // main1();
    // main2();
    main3();
}


fn main1() {
    println!("MAIN1");
    thread::spawn(|| {
        for i in 1..10 {
            println!("Hi number {} from the spawned thread", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Weirdly, if I don't put any other code in the main function,
    // the spawned threads wont fire?
    for i in 1..5 {
        println!("Hi number {} from the main thread", i);
        thread::sleep(Duration::from_millis(1));
    }

    // All the spawned threads will stop at the end of the main loop,
    // regardless if they're finished processing
}

fn main2() {
    println!("MAIN2");
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("Hi number {} from the spawned thread", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // We can join the handle to the main process,
    // which will pause the mainprocess until the handle is finished
    // handle.join().unwrap();

    for i in 1..5 {
        println!("Hi number {} from the main thread", i);
        thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap();
}

fn main3() {
    println!("MAIN3");
    let v = vec![1,2,3];

    // Note that the plain closure used takes no input or anything from the env
    // Because rust has no idea if the reference to env objects will last
    // What if the process outlives the variable??
    // ERROR:
    // let handle = thread::spawn(|| {
    //     for i in 1..10 {
    //         println!("Here's a vec: {:?}", v);
    //         thread::sleep(Duration::from_millis(1));
    //     }
    // });
    // FIX
    let handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        println!("Here's a vec: {:?}", v);
    });

    // Obviously the value has now been moved to the closure
    // so it's no longer accessible
    // ERROR
    // println!("Here's a vec: {:?}", v);

    handle.join().unwrap();
}
