"""Deploy the local sources or run a command on the processing server."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "run"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    host = os.environ["PROCESSING_HOST"]
    directory = shlex.quote(os.environ["PROCESSING_DIR"])
    connection = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host]

    if args.action == "deploy":
        subprocess.run(["git", "check-ignore", "--quiet", ".env"], cwd=ROOT, check=True)
        files = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
        ).split(b"\0")
        files = sorted({name for name in files if name and (ROOT / os.fsdecode(name)).is_file()})
        if (ROOT / ".env").is_file():
            files.append(b".env")
        archive = subprocess.Popen(
            ["tar", "-cf", "-", "--null", "-T", "-"], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        receiver = subprocess.Popen(
            connection + [f"umask 077; mkdir -p -- {directory} && tar -xf - -C {directory}"],
            stdin=archive.stdout,
        )
        archive.stdout.close()
        archive.stdin.write(b"\0".join(files) + b"\0")
        archive.stdin.close()
        receiver_status = receiver.wait()
        archive_status = archive.wait()
        if receiver_status or archive_status:
            raise SystemExit(receiver_status or archive_status)
        print("Deployed local sources and the ignored environment file.")
    else:
        if not args.command:
            parser.error("run requires a command")
        command = (
            f"cd -- {directory} && set -a && . ./.env && set +a && "
            'export PATH="$HOME/.local/bin:$PATH" && exec ' + shlex.join(args.command)
        )
        raise SystemExit(subprocess.call(connection + [command]))


if __name__ == "__main__":
    main()
