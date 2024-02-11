#!/usr/bin/env python
"""
Subshell spawner for Python

This utility searches the current directory for a virtual environment
in the current directory, travels up the directory tree (up to the
users home directory) until a virtual env is found, then spawns a
subshell that sources '.bashrc' and the virtual environment (and sets
a value for 'ENVY').

By default, searches for a virtual environment called '.venv', can be
specified in the first argument.

If a virtual environment is not found, a new virtual environment is
created at the current directory and a subshell sourcing the new venv
is invoked.

Different venvs can be targeted/created via first positional arg (e.g.
look for '.venv-foo' by using 'envy foo').
"""
from __future__ import annotations

import argparse
import os
import logging
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import pexpect
except ImportError:
    pexpect = None

VERSION = "1.0.0"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Shell:
    """Represents the current shell.

    A simplified version of the class from `python-poetry` that supports only non-windows systems

    https://github.com/python-poetry/poetry/blob/47a3a19c442c21aa5b2836ea092dda8ae4d13151/src/poetry/utils/shell.py
    """

    _shell = None

    def __init__(self, name: str, path: str) -> None:
        self._name = name
        self._path = path

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @classmethod
    def get(cls) -> Shell:
        """Retrieve the current shell."""
        if cls._shell is not None:
            return cls._shell

        if os.name == "posix":
            shell = os.environ.get("SHELL")
        elif os.name == "nt":
            shell = os.environ.get("COMSPEC")

        if not shell:
            raise RuntimeError("Unable to detect the current shell.")

        name, path = Path(shell).stem, shell

        cls._shell = cls(name, path)

        return cls._shell

    # TODO
    def activate(self, env_path: Path) -> Optional[int]:
        activate_script = self._get_activate_script()
        bin_dir = "bin"
        activate_path = env_path / bin_dir / activate_script

        if sys.platform == "win32":
            raise NotImplementedError("Not implemented for Windows platforms.")

        # c = subprocess.run(
        #     args=[self._path, "-i"],
        #     env=os.environ | {'ENVY': 'true'},
        # )
        terminal = shutil.get_terminal_size()
        c = pexpect.spawn(
            self._path,
            ["-i"],
            dimensions=(terminal.lines, terminal.columns),
            env=os.environ | {'ENVY': 'true'},
        )

        if self._name in ["zsh", "nu"]:
            c.setecho(False)

        if self._name == "zsh":
            # Under ZSH the source command should be invoked in zsh's bash emulator
            c.sendline(f"emulate bash -c '. {shlex.quote(str(activate_path))}'")
        elif self._name == "xonsh":
            c.sendline(f"vox activate {shlex.quote(str(env_path))}")
        else:
            cmd = f"{self._get_source_command()} {shlex.quote(str(activate_path))}"
            if self._name in ["fish", "nu"]:
                # Under fish and nu "\r" should be sent explicitly
                cmd += "\r"
            c.sendline(cmd)

        # def resize(sig: Any, data: Any) -> None:
        #     terminal = shutil.get_terminal_size()
        #     c.setwinsize(terminal.lines, terminal.columns)

        # signal.signal(signal.SIGWINCH, resize)

        # Interact with the new shell.
        c.interact(escape_character=None)
        c.close()

        sys.exit(c.exitstatus)

    def create_new_env(self, env_path: Path) -> int:
        if env_path.exists():
            raise FileExistsError(f"Path `{str(env_path)}` already exists.")
        c = subprocess.run(args=["python", "-m", "venv", env_path.name])
        return c.returncode



        ...

    def _get_activate_script(self) -> str:
        if self._name == "fish":
            suffix = ".fish"
        elif self._name in ("csh", "tcsh"):
            suffix = ".csh"
        elif self._name in ("powershell", "pwsh"):
            suffix = ".ps1"
        elif self._name == "cmd":
            suffix = ".bat"
        elif self._name == "nu":
            suffix = ".nu"
        else:
            suffix = ""

        return "activate" + suffix

    def _get_source_command(self) -> str:
        if self._name in ("fish", "csh", "tcsh"):
            return "source"
        elif self._name == "nu":
            return "overlay use"
        return "."

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self._name}", "{self._path}")'


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        help="Optional name of virtual environment to use (default: `.venv`).",
        type=str,
        nargs="?",
        default=".venv",
    )
    parser.add_argument(
        '-n',
        "--new",
        help="Create new virtualenv in current directory",
        action="store_true",
    )
    # TODO
    # parser.add_argument(
    #     '-U',
    #     "--update",
    #     help="Update script from Github."
    #     action="store_true",
    # )
    return parser.parse_args()

def is_venv_dir(path: Path, venv_name: str) -> bool:
    if (
            path.is_dir()
            and path.name == venv_name
            and ({'include', 'lib', 'share', 'lib64', 'etc', 'bin'} == {p.name for p in path.iterdir() if p.is_dir()})
    ):
        return True
    return False

def search_dirs_up(cwd: Path, venv_name: str) -> Optional[Path]:
    home = Path.home()
    for p in cwd.parents:
        logger.debug(f"Searching '{str(p)}'.")
        if p <= home:
            logger.debug(f"Selected path {str(p)} is in `home`, exiting search.")
            return None

        # Search and return if match
        for d in p.iterdir():
            if is_venv_dir(d, venv_name):
                return d

    return None

def check_deps():
    if pexpect is None:
        raise ImportError(
            "Module `pexpect` not found. Try install module before running again "
            f"(`$ pip install pexpect`) "
        )

def check_python_version():
    major, minor, *_ = version = platform.python_version_tuple()
    if int(major) <= 2 or int(minor) <= 5:
        raise EnvironmentError("Envy requires at least Python 3.5")

def main():
    check_deps()
    check_python_version()
    args = parse_cli()

    shell = Shell.get()
    cwd = Path.cwd()
    name = args.name

    # If envy is active, early return
    if os.environ.get("ENVY") is not None:
        raise EnvironmentError(
            f"Already set venv '{os.environ.get('VIRTUAL_ENV')}, "
            f"exit subshell to deactivate'."
        )

    # Create new venv if requested, early return
    if args.new:
        env_path = cwd / name
        result = shell.create_new_env(env_path)
        if result != 0:
            raise RuntimeError("Issue with creating venv :(")
        shell.activate(env_path)
        return

    # Search upwards for venv name
    env_path = search_dirs_up(cwd, name)
    if env_path is None:
        raise FileNotFoundError(
            f"Couldn't find venv named `{name}`, "
            f"try re-run with `--new` to create a new venv at current location"
        )

    shell.activate(env_path)


if __name__ == "__main__":
    try:
        logger.info("Executing")
        main()
    except Exception as err:
        logger.error(f"Error: {str(err)}")
        sys.exit(1)
    logger.info("Execution okay!")
    sys.exit(0)
