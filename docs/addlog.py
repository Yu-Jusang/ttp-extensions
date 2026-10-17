"""Append progress-log entries: python addlog.py entries.json [--status "..."] [--pop K]
entries.json: a UTF-8 JSON list of {date,time,tag,title,body} (or {"flow":[...], "files":[...]})."""
import json, sys, io, os
sys.stdout.reconfigure(encoding='utf-8')

here = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(here, 'log.json')
with io.open(log_path, encoding='utf-8') as f:
    d = json.load(f)
args = sys.argv[1:]
if '--pop' in args:
    k = int(args[args.index('--pop') + 1])
    d['entries'] = d['entries'][:-k]
if '--status' in args:
    d['status'] = args[args.index('--status') + 1]
if args and not args[0].startswith('--'):
    with io.open(args[0], encoding='utf-8') as f:
        new = json.load(f)
    if isinstance(new, dict):
        for key in ('flow', 'files', 'status'):
            if key in new:
                d[key] = new[key]
        new = new.get('entries', [])
    d['entries'].extend(new)
with io.open(log_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print(len(d['entries']), 'entries;', d['status'])
