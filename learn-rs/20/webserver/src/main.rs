// TODO
// [ ] Learn ablut TCP and HTTP
// [ ] Listen for TCP connection on a socket
// [ ] Parse a small number of HTTP requests
// [ ] Create a proper HTTP response
// [ ] Improve throughput with concurrency

use std::{
    fs,
    io::{prelude::*, BufReader},
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};

fn main() {
    // simple_tcp_demo();
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    for stream in listener.incoming() {
        let stream = stream.unwrap();
        // print_requests(stream);
        // handle_connection_basic(stream);
        // handle_connection_basic_html(stream);
        handle_connection(stream);

        // This is a dumb way to add multithreading:
        // thread::spawn(|| {
        //     handle_connection(stream);
        // });
        // Each new request will spawn a thread,
        // but if unlimited requests come in, this will crash the system
        // ... but it does solve the problem, let's build from here
    }
}

fn simple_tcp_demo() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();

    // `incoming()` returns an iterator of a sequence of tcp streams,
    // which blocks the program until a connection is made
    for stream in listener.incoming() {
        let stream = stream.unwrap();
        // A single stream is a connection between the client and the server
        // Running this program and navigating to the address in a browser
        // results in several prints, because the browser has a few streams
        println!("Conncection established!")
        // Also once the stream is out of scope,
        // The connection is dropped, so the browser may automatically retry
        // the connection
    }
}

fn print_requests(mut stream: TcpStream) -> TcpStream {
    let buf_reader = BufReader::new(&mut stream);
    let http_request: Vec<_> = buf_reader
        .lines()
        .map(|result| result.unwrap())
        .take_while(|line| !line.is_empty()) // End of request is signalled by \n\n
        .collect();

    println!("Request: {:#?}", http_request);
    stream
}

// Most of the requests look like this:
// ```
// Request: [
//     "GET / HTTP/1.1",
//     "Host: 127.0.0.1:7878",
//     "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:103.0) Gecko/20100101 Firefox/103.0",
//     "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
//     "Accept-Language: en-AU,en;q=0.5",
//     "Accept-Encoding: gzip, deflate, br",
//     "DNT: 1",
//     "Connection: keep-alive",
//     "Upgrade-Insecure-Requests: 1",
//     "Sec-Fetch-Dest: document",
//     "Sec-Fetch-Mode: navigate",
//     "Sec-Fetch-Site: none",
//     "Sec-Fetch-User: ?1",
// ]
// ```
// They're all `GET /` requests, meaning the browser is
// repeatedly trying to access the server

// HTTP is a text-based protocol, and takes the following format:
// ```
// Method Request-URI HTTP-Version CRLF
// headers CRLF
// message-body
// ```
// * `Method`: GET/POST etc
// * `Request-URI`: Uniform resource identify, target for request
// * `CRLF`: Carriage return line feed sequence, \r\n, separates request line and remainder

// Lets write a response of the following format:
// ```
// HTTP-Version Status-Code Reason-Phrase CRLF
// headers CRLF
// message-body
// ```
// In this case, it's probably sufficient to send back
// ```
// HTTP/1.1 200 OK\r\n\r\n
// ```

fn handle_connection_basic(mut stream: TcpStream) {
    // let mut stream = print_requests(stream);  // Looks like threes only one request now :)
    let response = "HTTP/1.1 200 OK\r\n\r\n";
    stream.write_all(response.as_bytes()).unwrap();
}

fn handle_connection_basic_html(mut stream: TcpStream) {
    let buf_reader = BufReader::new(&mut stream);
    let http_request: Vec<_> = buf_reader
        .lines()
        .map(|result| result.unwrap())
        .take_while(|line| !line.is_empty())
        .collect();

    let status_line = "HTTP/1.1 200 OK";
    let contents = fs::read_to_string("hello.html").unwrap();
    let crlf = "\r\n";
    let len = contents.len();

    let response = format!("{status_line}{crlf}Content-Length: {len}{crlf}{crlf}{contents}");
    stream.write_all(response.as_bytes()).unwrap();
}

// Lets now add a bit of request validation
fn handle_connection_draft(mut stream: TcpStream) {
    let buf_reader = BufReader::new(&mut stream);
    let request_line = buf_reader.lines().next().unwrap().unwrap();

    let crlf = "\r\n";
    println!("{}", request_line);
    if request_line == "GET / HTTP/1.1" {
        let status_line = "HTTP/1.1 200 OK";
        let contents = fs::read_to_string("hello.html").unwrap();
        let len = contents.len();
        let response = format!("{status_line}{crlf}Content-Length: {len}{crlf}{crlf}{contents}");
        stream.write_all(response.as_bytes()).unwrap();
    } else {
        let status_line = "HTTP/1.1 404 NOT FOUND";
        let contents = fs::read_to_string("404.html").unwrap();
        let len = contents.len();
        let response = format!("{status_line}{crlf}Content-Length: {len}{crlf}{crlf}{contents}");
        stream.write_all(response.as_bytes()).unwrap();
    }
}

// Now lets refactor
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

// Open two browser windows: one for http://127.0.0.1:7878/ and the other for
// http://127.0.0.1:7878/sleep. If you enter the / URI a few times, as before,
// you’ll see it respond quickly. But if you enter /sleep and then load /,
// you’ll see that / waits until sleep has slept for its full 5 seconds
// before loading.
//
// A possible solution to this is a thread pool..
