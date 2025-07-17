#!/usr/bin/env python

# echo "Hello there!" | nc -U -q 0 /tmp/pipe_viewer_socket

# curl -s -X POST 127.0.0.1:11434/api/generate --no-buffer -d '{"model": "phi4:latest", "prompt": "hello"}' -H 'Content-type: application/json' | while IFS= read -r line; do     echo "$line" | awk -F '"response":' '{print $2}' | awk -F ',"done":' '{print $1}' | sed 's/^"//;s/"$//' | tr -d '\n'; done | nc -U -q 0 /tmp/pipe_viewer_socket

# curl -s http://localhost:11434/api/chat --no-buffer -d '{
#   "model": "phi4:latest",
#   "messages": [
#     {
#       "role": "user",
#       "content": "hello"
#     }
#   ]
# }' | while IFS= read -r line; do
#     echo "$line" | awk -F '"response":' '{print $2}' | awk -F ',"done":' '{print $1}' | sed 's/^"//;s/"$//' | tr -d '\n';
# done |  nc -U -q 0 /tmp/pipe_viewer_socket

# curl -s http://localhost:11434/api/chat --no-buffer -d '{
#   "model": "phi4:latest",
#   "messages": [
#     {
#       "role": "user",
#       "content": "hello"
#     }
#   ]
# }' | while IFS= read -r line; do
#     echo "$line"
# done |  nc -U -q 0 /tmp/pipe_viewer_socket


# [
#   {
#     "role": "user",
#     "content": "hello"
#   },
#   {
#     "role": "assistant",
#     "content": "Hello! How can I assist you today?"
#   }
# ]

# echo '{"message":{"role": "user","content": "hello"}}' | nc -U -q 0 /tmp/pipe_viewer_socket

# TODO Parse messages


# cat /tmp/pipe_viewer_fifo

import time
import json
import os
import curses
import threading
import socket
import shlex
import textwrap
import subprocess
import re

SOCKET_PATH = "/tmp/pipe_viewer_socket"
FIFO_PATH_BUFFER = "/tmp/pipe_viewer_fifo_buffer"
FIFO_PATH_MESSAGES = "/tmp/pipe_viewer_fifo_messages"


class PipeViewer:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.running = True
        self.lock = threading.Lock()
        self.scroll_offset = 0
        self.pending_g = False  # Track if 'g' has been pressed once
        self.request_in_progress = False
        self.spinner_index = 0

        self.messages = []
        self.current_assistant_message = ""  # Track assistant response across multiple chunks

        # Ensure no leftover socket file
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        # Create a Unix domain socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(SOCKET_PATH)
        self.server_socket.listen()

        # Start the input listener in a separate thread
        self.listener_thread = threading.Thread(target=self.read_socket, daemon=True)
        self.listener_thread.start()

        # Create a FIFO for output
        self.fifo_buffer_path = FIFO_PATH_BUFFER
        if not os.path.exists(self.fifo_buffer_path):
            os.mkfifo(self.fifo_buffer_path)

        # Start the output transmitter in a separate thread
        self.fifo_buffer_thread = threading.Thread(target=self.write_to_fifo_buffer, daemon=True)
        self.fifo_buffer_thread.start()

        # Create a FIFO for output messages
        self.fifo_messages_path = FIFO_PATH_MESSAGES
        if not os.path.exists(self.fifo_messages_path):
            os.mkfifo(self.fifo_messages_path)

        # Start the output transmitter in a separate thread
        self.fifo_messages_thread = threading.Thread(
            target=self.write_to_fifo_messages, daemon=True
        )
        self.fifo_messages_thread.start()

        self.run()

    def read_socket(self):
        """Continuously accept connections and read from the Unix domain socket."""
        try:
            while self.running:
                conn, _ = self.server_socket.accept()
                with conn:
                    while self.running:
                        data = conn.recv(1024)
                        if not data:
                            break
                        with self.lock:
                            decoded_data = data.decode()
                            try:
                                self.handle_input(decoded_data)
                            except json.JSONDecodeError:
                                continue

        except KeyboardInterrupt:
            self.running = False

    def handle_input(self, line):
        data = json.loads(line, strict=False)
        role = data.get("message", {}).get("role")
        content = data.get("message", {}).get("content", "")
        done = data.get("done", False)

        if role == "user":
            if not self.request_in_progress:
                self.messages.append({"role": "user", "content": content})
                self.messages.append({"role": "assistant", "content": ""})
                self.request_in_progress = True
                self.submit_request(content)
            else:
                raise ValueError("Foo")
            content += "\n\n"
        elif role == "assistant":
            self.current_assistant_message += content  # Append incoming text
            self.messages[-1]["content"] = self.current_assistant_message
            if done:
                self.current_assistant_message = ""  # Reset for next response
                self.request_in_progress = False
                content += "\n\n"

        return content

    def submit_request(self, user_content):
        """
        Automatically call the curl command with the provided user content.
        The command will stream its output back to our socket.
        """
        payload = {
            "model": "phi4:latest",
            "messages": self.messages,
        }
        # Create a JSON string payload.
        json_payload = json.dumps(payload)
        safe_payload = shlex.quote(json_payload)  # Properly escapes the payload

        # Build the command string.
        # Note: We use double quotes outside and single quotes inside the JSON payload.
        command = (
            f"curl -s http://localhost:11434/api/chat --no-buffer -d {safe_payload} | "
            'while IFS= read -r line; do echo "$line"; done | nc -U -q 0 /tmp/pipe_viewer_socket'
        )
        # Launch the command asynchronously.
        subprocess.Popen(command, shell=True)

    def write_to_fifo_buffer(self):
        """Continuously write buffer to FIFO for external access."""
        while self.running:
            h, w = self.stdscr.getmaxyx()
            with open(self.fifo_buffer_path, "w") as fifo:
                with self.lock:
                    fifo.write("\n".join(self.get_wrapped_buffer(w)) + "\n")
            time.sleep(1)  # Adjust update frequency

    # def write_to_fifo_messages(self):
    #     """Continuously write buffer to FIFO for external access."""
    #     while self.running:
    #         with open(self.fifo_messages_path, "w") as fifo:
    #             with self.lock:
    #                 fifo.write(json.dumps(self.messages, indent=2) + "\n")
    #         time.sleep(1)  # Adjust update frequency
    def write_to_fifo_messages(self):
        """Continuously write buffer to FIFO for external access."""
        with open(self.fifo_messages_path, "w") as fifo:  # Open once, keep open
            while self.running:
                with self.lock:
                    fifo.write(json.dumps(self.messages, indent=2) + "\n")
                    fifo.flush()  # Ensure data is written immediately
                time.sleep(1)  # Adjust update frequency

    def get_wrapped_buffer(self, w):
        wrapped_buffer = []
        # Process each message in your history.
        for msg in self.messages:
            if msg['role'] == "user":
                wrapped_buffer.extend([">>>> 👤 USER:", ""])
            elif msg['role'] == "assistant":
                wrapped_buffer.extend([">>>> 🤖 ASSISTANT:", ""])
            in_code_block = False  # Flag to indicate we're in a code block.
            for line in msg["content"].splitlines():
                # Check for code block delimiters.
                if line.strip().startswith("```"):
                    wrapped_buffer.append(line)  # Always add the delimiter as is.
                    in_code_block = not in_code_block  # Toggle state.
                    continue
                # If we're in a code block, don't wrap.
                if in_code_block:
                    wrapped_buffer.append(line)
                # If the line is empty or consists only of whitespace, just add it.
                elif not line or re.match(r"\s+", line):
                    wrapped_buffer.append(line)
                else:
                    # Wrap the line using textwrap.
                    wrapped_buffer.extend(
                        textwrap.wrap(
                            line,
                            width=w - 1,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                    )
            wrapped_buffer.extend(["", "<<<<<<<<", "", ""])
        return wrapped_buffer

    def run(self):
        """Main curses loop to update the display."""
        curses.curs_set(0)  # Hide cursor
        self.stdscr.nodelay(1)  # Make getch non-blocking
        self.stdscr.timeout(100)  # Refresh every 100ms
        spinner_chars = list(reversed("⣾⣽⣻⢿⡿⣟⣯⣷"))
        while self.running:
            self.stdscr.clear()
            h, w = self.stdscr.getmaxyx()

            with self.lock:
                wrapped_buffer = self.get_wrapped_buffer(w)

                total_lines = len(wrapped_buffer)
                start = max(0, total_lines - h - self.scroll_offset)
                visible_lines = wrapped_buffer[start : start + h]

                for i, line in enumerate(visible_lines):
                    if i >= h:
                        break
                    self.stdscr.addnstr(i, 0, line, w - 1)

            # If a request is in progress, draw a spinner.
            if self.request_in_progress:
                spinner_char = spinner_chars[self.spinner_index % len(spinner_chars)]
                self.spinner_index += 1
                # Draw the spinner at the bottom right corner.
                try:
                    self.stdscr.addnstr(h - 1, 1, f"{spinner_char} Generating response", w - 1)
                except curses.error:
                    pass

            self.stdscr.refresh()
            key = self.stdscr.getch()
            if key == ord("q"):
                break
            elif key == ord("j"):
                self.scroll_offset = max(self.scroll_offset - 1, 0)
            elif key == ord("k"):
                self.scroll_offset = min(self.scroll_offset + 1, max(0, total_lines - h))
            elif key == 4:  # Ctrl-D
                self.scroll_offset = max(self.scroll_offset - h // 2, 0)
            elif key == 21:  # Ctrl-U
                self.scroll_offset = min(self.scroll_offset + h // 2, max(0, total_lines - h))
            elif key == 6:  # Ctrl-F
                self.scroll_offset = max(self.scroll_offset - h, 0)
            elif key == 2:  # Ctrl-B
                self.scroll_offset = min(self.scroll_offset + h, max(0, total_lines - h))
            elif key == ord("g"):
                if self.pending_g:
                    self.scroll_offset = max(0, total_lines - h)  # Go to top
                    self.pending_g = False
                else:
                    self.pending_g = True  # Wait for another 'g'
            elif key == ord("G"):
                self.scroll_offset = 0  # Go to bottom
            elif key == curses.KEY_RESIZE:
                curses.update_lines_cols()
                h, w = self.stdscr.getmaxyx()
                self.scroll_offset = max(
                    0, len(wrapped_buffer) - h
                )  # Adjust scroll position after resize
            else:
                self.pending_g = False  # Reset if any other key is pressed

        self.running = False
        self.server_socket.close()
        os.remove(SOCKET_PATH)


if __name__ == "__main__":
    try:
        curses.wrapper(PipeViewer)
    except KeyboardInterrupt:
        pass
