
import json
import os
import threading
import socket
import shlex
import subprocess


SOCKET_PATH = "/tmp/pipe_viewer_socket"
FIFO_PATH_BUFFER = "/tmp/pipe_viewer_fifo_buffer"
FIFO_PATH_MESSAGES = "/tmp/pipe_viewer_fifo_messages"

class PipeViewer:
    def __init__(self):
        self.buffer = ""  # []
        self.running = True
        self.lock = threading.Lock()
        self.scroll_offset = 0
        self.pending_g = False  # Track if 'g' has been pressed once
        self.request_in_progress = False

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
                            # print(repr(data))
                            decoded_data = data.decode("utf-8")
                            try:
                                content = self.handle_input(decoded_data)
                            except json.JSONDecodeError:
                                continue

                            # TODO fixnewlines in response
                            self.buffer += content
                            # if self.buffer and not self.buffer[-1].endswith("\n"):
                            #     last_line = self.buffer.pop()
                            #     content = last_line + content

                            # self.buffer.extend(content.splitlines())

                            # if self.scroll_offset == 0:  # Auto-scroll only if at the bottom
                            #     self.scroll_offset = max(
                            #         0, len(self.buffer) - self.stdscr.getmaxyx()[0]
                            #     )
        except KeyboardInterrupt:
            self.running = False

    def handle_input(self, line):
        data = json.loads(line, strict=False)
        message = data["message"]
        print(repr(message))
        role = message["role"]
        content = message["content"]
        done = data.get("done", False)

        if role == "user":
            if not self.request_in_progress:
                self.messages.append({"role": "user", "content": content})
                self.request_in_progress = True
                self.submit_request(content)
            else:
                raise ValueError("Foo")
            content += "\n\n\n\n"
        elif role == "assistant":
            self.current_assistant_message += content  # Append incoming text
            if done:
                # Finalize assistant message
                self.messages.append(
                    {"role": "assistant", "content": self.current_assistant_message}
                )
                self.current_assistant_message = ""  # Reset for next response
                self.request_in_progress = False
                content += "\n\n\n\n"

        return content.replace("\n", "\n\n\n\n")

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
        """Main curses loop to update the display."""

        while self.running:
            True

        self.running = False
        self.server_socket.close()
        os.remove(SOCKET_PATH)


if __name__ == "__main__":
    print("RUNNNING")
    PipeViewer().run()
