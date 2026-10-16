"""Read-only official entry check. Never print or persist credentials/API error bodies."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT/'toolkit/8d2bce64/v2d_submission_kit'
REPORT = ROOT/'notes/track3-kaggle-entry.json'


def main():
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'track': 'track_3',
              'read_only': True, 'uploads_performed': 0, 'rules_accepted_by_script': False,
              'credentials_logged': False, 'competitions': []}
    hidden = io.StringIO()
    try:
        with redirect_stdout(hidden), redirect_stderr(hidden):
            spec = importlib.util.spec_from_file_location('v2d_official_submit', KIT/'submit/v2d_submit.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            config = json.loads((KIT/'config/tracks.json').read_text()); track = config['tracks']['track_3']
            competitions = dict(track['competitions'])
            if track.get('register_object_mesh') and 'track_3_mesh' in config['tracks']:
                competitions.update(config['tracks']['track_3_mesh']['competitions'])
            api = module.build_api()
            for metric, slug in competitions.items():
                entered = module.entry_status(api, slug)
                report['competitions'].append({'metric': metric, 'slug': slug,
                    'entered': entered if entered in (True, False, None) and entered is not module.MISSING else None,
                    'status': 'missing_competition' if entered is module.MISSING else
                              ('entered' if entered is True else 'needs_rules' if entered is False else 'unknown'),
                    'rules_url': f'https://www.kaggle.com/competitions/{slug}/rules'})
        states = [c['status'] for c in report['competitions']]
        report.update(all_six_entered=len(states)==6 and all(s=='entered' for s in states),
                      state='verified' if len(states)==6 and all(s=='entered' for s in states) else
                            'needs_rules' if 'needs_rules' in states else 'unverified')
    except (Exception, SystemExit) as exc:
        report.update(state='authentication_or_client_failed', error_type=type(exc).__name__,
                      all_six_entered=False,
                      detail='Client failed before a complete entry check; raw exception and output intentionally omitted.')
    finally:
        hidden.close()
    REPORT.parent.mkdir(exist_ok=True)
    temp = REPORT.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)+'\n'); temp.replace(REPORT)
    print(json.dumps(report, indent=2))
    return 0 if report.get('all_six_entered') else 1


if __name__ == '__main__':
    raise SystemExit(main())
