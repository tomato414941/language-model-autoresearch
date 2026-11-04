import datetime
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / 'results' / 'operations'

def api(method, path, data=None, params=None):
    token = os.environ['VAST_API_KEY'].strip()
    version = 'v1' if path.startswith('instances') else 'v0'
    url = 'https://cloud.vast.ai/api/' + version + '/' + path.lstrip('/')
    if params:
        url += '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), method=method, headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        print('API error:', error.code, error.read().decode()[:2000], file=sys.stderr)
        raise

def compact(row, fields):
    return {key: row.get(key) for key in fields if key in row}

def main():
    os.umask(0o077)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    command = sys.argv[1]
    if command == 'audit':
        instances = api('GET', 'instances/', params={'owner': 'me'})
        print('Instances:', json.dumps([compact(row, ['id', 'label', 'gpu_name', 'num_gpus', 'actual_status', 'start_date', 'end_date', 'duration', 'dph_total']) for row in instances.get('instances', [])]))
        now = datetime.datetime.now(datetime.timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        result = api('GET', 'users/me/invoices', params={'owner': 'me', 'sdate': start, 'edate': time.time(), 'inc_charges': 'true'})
        (STATE_DIR / 'billing-audit.json').write_text(json.dumps(result, indent=2))
        print('Billing keys:', list(result))
        rows = result.get('invoices', [])
        print('Today billing:', json.dumps([compact(row, ['id', 'instance_id', 'label', 'description', 'type', 'amount', 'timestamp', 'start_time', 'end_time']) for row in rows]))
    elif command == 'search':
        query = {'verified': {'eq': True}, 'external': {'eq': False}, 'rentable': {'eq': True}, 'rented': {'eq': False}, 'num_gpus': {'eq': 1}, 'gpu_name': {'in': ['H100 SXM', 'H100 NVL', 'H100 PCIE']}, 'gpu_ram': {'gte': 79000}, 'cpu_ram': {'gte': 32000}, 'cpu_cores_effective': {'gte': 12}, 'disk_space': {'gte': 90}, 'inet_down': {'gte': 800}, 'reliability2': {'gte': 0.98}, 'cuda_max_good': {'gte': 12.8}, 'order': [['dph_total', 'asc']], 'type': 'ondemand', 'allocated_storage': 90, 'limit': 20}
        result = api('POST', 'bundles/', query)
        (STATE_DIR / 'offers.json').write_text(json.dumps(result, indent=2))
        print(json.dumps([compact(row, ['id', 'gpu_name', 'num_gpus', 'gpu_ram', 'dph_total', 'inet_down', 'inet_down_cost', 'cpu_cores_effective', 'cpu_ram', 'cuda_max_good', 'driver_version', 'disk_space', 'reliability2', 'dlperf', 'duration', 'geolocation', 'storage_cost']) for row in result.get('offers', [])], indent=2))
    else:
        raise SystemExit('Unknown command')

if __name__ == '__main__':
    main()
