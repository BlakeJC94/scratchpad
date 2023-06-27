// TODO
// [X] Learn ablut TCP and HTTP
// [X] Listen for TCP connection on a socket
// [X] Parse a small number of HTTP requests
// [X] Create a proper HTTP response
// [ ] Improve throughput with concurrency

use std::{
    fs,
    io::{prelude::*, BufReader},
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};
use webserver::ThreadPool;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    let pool = ThreadPool::new(4);

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        pool.execute(|| {
            handle_connection(stream);
        });

    }
}

fn handle_connection(mut stream: TcpStream) {
    let buf_reader = BufReader::new(&mut stream);
    let request_line = buf_reader.lines().next().unwrap().unwrap();

    // Here let's simulate a slow connection on the page /sleep
    let (status_line, filename) = match &request_line[..] {
        "GET / HTTP/1.1" => ("HTTP/1.1 200 OK", "hello.html"),
        "GET /sleep HTTP/1.1" => {  // Browser hangs on this page :(
            thread::sleep(Duration::from_secs(5));
            ("HTTP/1.1 200 OK", "hello.html")
        },
        _ => ("HTTP/1.1 404 NOT FOUND", "404.html"),
    };

    let contents = fs::read_to_string(filename).unwrap();
    let len = contents.len();

    let crlf = "\r\n";
    let response = format!("{status_line}{crlf}Content-Length: {len}{crlf}{crlf}{contents}");

    stream.write_all(response.as_bytes()).unwrap();
}
