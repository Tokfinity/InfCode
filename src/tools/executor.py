import subprocess
import uuid
import docker
import pexpect
import re
from docker.errors import DockerException, ImageNotFound, NotFound
from src.managers.log.logger import Logger


class Executor:
    def __init__(self, image: str, logger: Logger | None = None):
        self.image = image
        self.container = None
        self.sessions: dict[str, pexpect.spawn] = {}
        self.client = docker.from_env()
        self.logger = logger

        try:
            self.client.images.get(self.image)
        except ImageNotFound:
            raise DockerException(
                f"Image '{self.image}' not found. Please build the image first."
            )

        try:
            self.container = self.client.containers.run(
                self.image,
                command="sleep infinity",
                detach=True,
                working_dir="/workspace",
            )
            self.logger.info(f"Created container {self.container.id}")
        except DockerException as e:
            raise DockerException(
                f"Failed to create container with image '{self.image}': {e}"
            )

        session_id = self.init_session()
        if session_id is None:
            raise DockerException("Failed to initialize default session")
        if session_id in self.sessions:
            self.sessions["0"] = self.sessions.pop(session_id)

    def init_session(self) -> str:
        session_id = str(uuid.uuid4())
        command = f"docker exec -it {self.container.id} /bin/bash"

        for attempt in range(3):  # Retry up to 3 times
            try:
                shell = pexpect.spawn(command, encoding="utf-8", timeout=120)
                shell.expect([r"\$.*", r"#.*"], timeout=120)

                # Source conda and activate testbed environment
                shell.sendline("source /opt/miniconda3/bin/activate")
                shell.expect([r"\$.*", r"#.*"], timeout=30)

                shell.sendline("conda activate testbed")
                shell.expect([r"\$.*", r"#.*"], timeout=30)

                shell.sendline("export NO_COLOR=1 && export PAGER=cat")
                shell.expect([r"\$.*", r"#.*"], timeout=30)

                # Verify conda environment is alive by checking the full output
                # The output should contain (testbed) if the environment is activated
                # We can check this by looking at the full output from the conda activate command
                output = shell.before
                if "(testbed)" not in output:
                    # Environment not properly activated, retry
                    if attempt < 2:  # Not the last attempt
                        shell.close(force=True)
                        continue
                    else:
                        shell.close(force=True)
                        raise DockerException(
                            "Failed to activate conda environment 'testbed' after 3 attempts"
                        )

                self.sessions[session_id] = shell
                return session_id

            except pexpect.exceptions.TIMEOUT:
                if attempt < 2:  # Not the last attempt
                    if "shell" in locals() and shell.isalive():
                        shell.close(force=True)
                    continue
                else:
                    return None
            except Exception as e:
                if attempt < 2:  # Not the last attempt
                    if "shell" in locals() and shell.isalive():
                        shell.close(force=True)
                    continue
                else:
                    raise DockerException(
                        f"Failed to initialize session after 3 attempts: {e}"
                    )

        return None

    def execute(
        self, session_id: str, command: str, timeout: int = 300
    ) -> tuple[int, str]:
        shell = self.sessions.get(session_id)
        if not shell or not shell.isalive():
            return -1, "Session not found or is dead."

        full_command = command.strip()
        shell.sendline(full_command)
        marker = f"---CMD_DONE---"
        marker_command = f"echo {marker}$?"
        shell.sendline(marker_command)
        try:
            shell.expect(marker + r"(\d+).*[\n](.*)", timeout=timeout)
        except pexpect.exceptions.TIMEOUT:
            return (
                -1,
                f"Error: Command '{command}' timed out after {timeout} seconds. Partial output:\n{shell.before}",
            )
        exit_code = int(shell.match.group(1))
        p = str(shell.match.group(2))
        all_lines: str = p + shell.before
        # delete all \r
        all_lines = re.sub(r"\r", "", all_lines)
        # Remove some non-color-related terminal control characters.
        # \x1b[?2004h - tell terminal to activate special paste process
        # \x1b[?2004l - tell terminal to activate special paste process
        all_lines = re.sub(r"\x1B\[\?2004[l|h]", "", all_lines)
        # Strip the last line's echo.
        all_lines = re.sub(r"\n[^\n]+---CMD_DONE---.*", "", all_lines)
        # self.logger.info(f"'{[all_lines]}'")
        return exit_code, all_lines

    def execute_once(self, command: str, timeout: int = 300) -> tuple[int, str]:
        # cmd = ["docker", "exec", self.container.id, "bash", "-c", command]
        cmd = ["docker", "exec", "-i", self.container.id, "bash", "-s"]
        sub = subprocess.run(
            cmd,
            encoding="utf-8",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=f"{command}\n",
        )
        if sub.returncode != 0:
            return sub.returncode, sub.stderr
        return sub.returncode, sub.stdout

    def cpfile_host_to_container(self, source: str, dest: str) -> tuple[int, str]:
        cmd = ["docker", "cp", source, f"{self.container.id}:{dest}"]
        sub = subprocess.run(
            cmd,
            encoding="utf-8",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.execute_once(f"chmod 0777 {dest}")
        if sub.returncode != 0:
            return sub.returncode, sub.stderr
        return sub.returncode, sub.stdout

    def check_session(self, session_id: str = "0") -> bool:
        """
        Check whether the current '0' session is alive and restart it if not.
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session and session.isalive():
                return True
            else:
                self.sessions.pop(session_id)

        new_session_id = self.init_session()
        if new_session_id is None:
            return False
        if new_session_id != session_id:
            self.sessions[session_id] = self.sessions.pop(new_session_id)

        return True

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            if session and session.isalive():
                session.close(force=True)
        # Session not found - this is not an error condition

    def shutdown(self):
        for session_id in list(self.sessions.keys()):
            self.close_session(session_id)

        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except DockerException as e:
                pass  # Silently handle cleanup errors
        self.container = None
