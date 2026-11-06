"""Read the running job's GPU state, process handle and log tails without restarting work."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.rental import load, ssh_command
from tools import vast_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identifier')
    args = parser.parse_args()
    job = load(args.identifier)
    returned = {row['instance'] for row in vast_backend.entries() if row.get('event') == 'returned'}
    held = job.get('instance_id') is not None and job['instance_id'] not in returned
    print(json.dumps(dict(
        observed_at=dt.datetime.now(dt.timezone.utc).isoformat(), phase=job['phase'],
        expires_at=job.get('expires_at'), exit_code=job.get('exit_code'),
        instance=job.get('instance_id'), returned=not held,
    ), indent=2), flush=True)
    if not held or not job.get('ssh_config'):
        return
    with tempfile.TemporaryDirectory(prefix='autoresearch-observe-') as temporary:
        key = Path(temporary) / 'key'
        key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
        key.chmod(0o600)
        command = (
            'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; '
            'if test -d autoresearch; then cd autoresearch; '
            'if test -f run.pid; then printf "run_pid: "; cat run.pid; '
            'ps -p "$(cat run.pid)" -o pid=,etime=,stat=,args=; fi; '
            'if test -f run.exit; then printf "run_exit: "; cat run.exit; fi; '
            'for file in bootstrap.log prepare.log; do '
            'if test -f "$file"; then printf "\\n%s\\n" "$file"; tail -n 4 "$file"; fi; done; '
            'if test -f run.log; then printf "\\nrun.log\\n"; '
            'tail -c 6000 run.log | tr "\\r" "\\n" | tail -n 8; fi; fi'
        )
        result = subprocess.run(ssh_command(job, key) + [command],
                                capture_output=True, text=True, timeout=30)
        print(result.stdout.replace('\r', '\n'), end='', flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr, end='')
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
