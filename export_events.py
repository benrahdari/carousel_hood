"""Export de-identified event and session tables from the IUI'25 CARE movie study.
usage: python -I export_events.py <raw dir> <out dir>
- participant ids: P### in sorted raw-id order (same as build.py); raw platform ids are dropped
- times: seconds from the participant's first logged event; calendar dates dropped
- keydown: only non-character keys kept (character keys -> 'char')
- surveys: age band, digital comfort, internet usage (coded as in the raw file)"""
import json, glob, sys, csv, os
from datetime import datetime
ts = lambda t: datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()
raw, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(raw + '/*.json'))
ev_rows, ses_rows, rec_rows = [], [], []
for pi, p in enumerate(files):
    d = json.load(open(p)); pid = 'P%03d' % pi; sp = d['session_parameters']
    ev = sorted(d['interactions'], key=lambda e: e['timestamp'])
    t0 = ts(ev[0]['timestamp']) if ev else ts(d['start_time'])
    for e in ev:
        x = e['details']; key = x.get('key')
        if key is not None and len(key) == 1: key = 'char'
        mid = x.get('movieId'); mid = '' if mid in (None, 'not_set', '') else mid
        ev_rows.append(dict(participant=pid, t=round(ts(e['timestamp']) - t0, 3), event=e['eventType'],
                            x=x.get('x', ''), y=x.get('y', ''), scroll_x=x.get('scrollX', ''), scroll_y=x.get('scrollY', ''),
                            movieId=mid, row_position=x.get('row_position', ''), col_position=x.get('col_position', ''),
                            row_genre=x.get('genre', ''), on_movie=x.get('click_on_movie', ''),
                            target_id=x.get('orig_target_id', ''), target_class=x.get('orig_target_className', ''), key=key or ''))
    s = None
    sf = os.path.join(raw, 'survey', f"{d['user_id']}_survey.json")
    if os.path.exists(sf): s = json.load(open(sf))
    ses_rows.append(dict(participant=pid, adaptation=sp['type'], baseline=sp['baseline'], ui=sp['ui'],
                         mouse_logging_ms=sp.get('mouse_logging_freq', ''), n_events=len(ev),
                         session_secs=round(ts(ev[-1]['timestamp']) - t0, 1) if ev else '',
                         age_band=(s or {}).get('age', ''), digital_comfort=(s or {}).get('digital_comfort', ''),
                         internet_usage=(s or {}).get('internet_usage', '')))
    r = d['recommendations']; groups = r.items() if isinstance(r, dict) else [('list', r)]
    for gname, lst in groups:
        for rank, m in enumerate(lst):
            rec_rows.append(dict(participant=pid, group=gname, rank=rank, movieId=m.get('movieId'), title=m.get('Title'),
                                 year=m.get('Year'), genres=m.get('genres')))
for name, rows in (('events', ev_rows), ('sessions', ses_rows), ('shown_titles', rec_rows)):
    with open(os.path.join(out, name + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(name, len(rows))
