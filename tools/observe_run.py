"""Read the existing rental's GPU state, process handle and log tails without restarting work."""

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
from tools.rental import load, ssh_connection
from tools.vast_control import list_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identifier')
    args = parser.parse_args()
    job = load(args.identifier)
    matches = [row for row in list_instances(timeout=15)
               if row.get('label') == job['label'] and row['id'] == job.get('instance_id')]
    print(json.dumps(dict(
        observed_at=dt.datetime.now(dt.timezone.utc).isoformat(), phase=job['phase'],
        expires_at=job.get('expires_at'), exit_code=job.get('exit_code'),
        instances=[{key: row.get(key) for key in ('id', 'gpu_name', 'num_gpus', 'actual_status')}
                   for row in matches],
    ), indent=2), flush=True)
    if not matches:
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
            'for file in bootstrap.log prepare.log run.log; do '
            'if test -f "$file"; then printf "\\n%s\\n" "$file"; tail -c 4000 "$file"; fi; done; fi'
        )
        result = subprocess.run(ssh_connection(matches[0], key) + [command],
                                capture_output=True, text=True, timeout=30)
        print(result.stdout.replace('\r', '\n'), end='', flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr, end='')
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
