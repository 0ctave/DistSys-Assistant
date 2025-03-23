import os
import subprocess

class CommandExecutor:

    def __init__(self, password):

        self._password = password

        home_dir = os.path.expanduser("~")
        self._shell_process = subprocess.Popen(
            ["/bin/bash", "-i"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd = home_dir,
            start_new_session=True
        )

        # Send initial setup command to ensure clean output
        self._shell_process.stdin.write("export PS1='' && echo SHELL_READY\n")
        self._shell_process.stdin.flush()

        # Read until we confirm shell is ready
        while True:
            line = self._shell_process.stdout.readline().strip()
            if line == "SHELL_READY":
                break  # Shell is ready to receive commands

    def run_command(self, command):
        """Runs a command in the persistent shell and returns its output."""
        end_marker = "CMD_DONE"

        if command.strip().startswith("sudo"):
            # Add -S to allow sudo to read from stdin
            command = command.replace("sudo", "sudo -S", 1)
            self._shell_process.stdin.write(f"echo '{self._password}' | {command} ; echo {end_marker}\n")
        else:
            self._shell_process.stdin.write(f"{command} ; echo {end_marker}\n")

        self._shell_process.stdin.flush()

        output_lines = []
        while True:
            line = self._shell_process.stdout.readline().strip()
            if line == end_marker:
                break  # Command has finished
            output_lines.append(line)

        return "\n".join([line.strip() for line in output_lines if end_marker not in line])

    def close(self):
        """Closes the persistent shell process."""
        self._shell_process.stdin.close()
        self._shell_process.terminate()
        self._shell_process.wait()