#!/usr/bin/env python

# echo "Hello there!" | nc -U -q 0 /tmp/pipe_viewer_socket

# curl -s -X POST 127.0.0.1:11434/api/generate --no-buffer -d '{"model": "phi4:latest", "prompt": "hello"}' -H 'Content-type: application/json' | while IFS= read -r line; do     echo "$line" | awk -F '"response":' '{print $2}' | awk -F ',"done":' '{print $1}' | sed 's/^"//;s/"$//' | tr -d '\n'; done | nc -U -q 0 /tmp/pipe_viewer_socket

curl -s http://localhost:11434/api/chat --no-buffer -d '{
  "model": "phi4:latest",
  "messages": [
    {
        "role": "system",
        "content": "you are a pirate"
    },
    {
      "role": "user",
      "content": "hello"
    }
  ]
}' | while IFS= read -r line; do
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


# cat /tmp/pipe_viewer_fifo

import json
import os
import time
import threading
import socket
import shlex
import textwrap
import subprocess
import re
import queue
from collections import deque

SOCKET_PATH = "/tmp/pipe_viewer_socket"
FIFO_PATH_BUFFER = "/tmp/pipe_viewer_fifo_buffer"
FIFO_PATH_MESSAGES = "/tmp/pipe_viewer_fifo_messages"


class PipeViewer:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.request_in_progress = False
        self.spinner_index = 0

        self.messages = []
        self.current_assistant_message = ""  # Track assistant response across multiple chunks
        self.buffer = ""
        self.message_queue = queue.Queue()  # The queue to store messages

        # Ensure no leftover socket file
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        # Create a Unix domain socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(SOCKET_PATH)
        self.server_socket.listen()

        # Create a FIFO for output messages
        self.fifo_buffer_path = FIFO_PATH_BUFFER
        if not os.path.exists(self.fifo_buffer_path):
            os.mkfifo(self.fifo_buffer_path)

        # Create a FIFO for output messages
        # self.fifo_messages_path = FIFO_PATH_MESSAGES
        # if not os.path.exists(self.fifo_messages_path):
        #     os.mkfifo(self.fifo_messages_path)

        # Start the input listener in a separate thread
        self.listener_thread = threading.Thread(target=self.read_socket, daemon=True)
        self.listener_thread.start()

        # Start the output transmitter in a separate thread
        self.fifo_messages_thread = threading.Thread(
            target=self.write_to_fifo_buffer, daemon=True
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
                                content = self.handle_input(decoded_data)
                            except json.JSONDecodeError:
                                continue
                            self.message_queue.put(content)

        except KeyboardInterrupt:
            self.running = False

    def handle_input(self, line):
        data = json.loads(line, strict=False)
        role = data.get("message", {}).get("role")
        content = data.get("message", {}).get("content", "")
        done = data.get("done", False)

        if role == "user":
            content = ">>>> 👤 USER:\n\n" + content
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


    def write_to_fifo_buffer(self):
        """Continuously read from the message queue and write to the FIFO."""
        try:
            buffer_len = 0
            in_code_block = False
            msg_buffer = deque()
            with open(self.fifo_buffer_path, "ab", buffering=0) as fifo:  # Open FIFO once, keep open
                while self.running:
                    if not self.message_queue.empty():
                        message = self.message_queue.get()

                        if len(msg_buffer) > 3:
                            msg_buffer.popleft()
                        msg_buffer.append(message)

                        if not in_code_block and buffer_len + len(message.split('\n')[0]) > 80:
                            fifo.write("\n".encode())
                            buffer_len = 0

                        fifo.write(message.encode())
                        fifo.flush()  # Ensure data is written immediately

                        if '\n' in message:
                            buffer_len = len(message.split('\n')[-1])
                        else:
                            buffer_len += len(message)

                        if '```' in ''.join([str(i) for i in msg_buffer]):
                            msg_buffer.clear()
                            in_code_block = not in_code_block

                    time.sleep(0.001)  # Adjust as needed to prevent CPU overload

        except KeyboardInterrupt:
            self.running = False

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

    def run(self):
        # spinner_chars = list(reversed("⣾⣽⣻⢿⡿⣟⣯⣷"))
        while self.running:
            pass
        self.running = False
        self.server_socket.close()
        os.remove(SOCKET_PATH)
        os.remove(FIFO_PATH_BUFFER)
        os.remove(FIFO_PATH_MESSAGES)

    # def write_to_fifo_messages(self):
    #     """Continuously write buffer to FIFO for external access."""
    #     with open(self.fifo_messages_path, "w") as fifo:  # Open once, keep open
    #         while self.running:
    #             with self.lock:
    #                 fifo.write(json.dumps(self.messages, indent=2) + "\n")
    #                 fifo.flush()  # Ensure data is written immediately
    #             time.sleep(1)  # Adjust update frequency

    def get_wrapped_buffer(self, w):
        wrapped_buffer = []
        # Process each message in your history.
        for msg in self.messages:
            if msg["role"] == "user":
                wrapped_buffer.extend([">>>> 👤 USER:", ""])
            elif msg["role"] == "assistant":
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


if __name__ == "__main__":
    try:
        PipeViewer().run()
    except KeyboardInterrupt:
        pass

