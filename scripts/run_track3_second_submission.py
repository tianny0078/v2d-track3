"""Isolate submission #2 state/receipt while reusing the first coordinator's logic.

Preparation/inspection is the default. This module never starts an upload merely
because it is imported, inspected, or launched without --submit-on-ready.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import run_track3_submission_watch as watch

ROOT = Path(__file__).resolve().parents[1]
FIRST_STATUS = ROOT/'notes/track3-submission-status.json'


def separate(cohort, status, job):
    first = watch.read(FIRST_STATUS, {})
    assert status != FIRST_STATUS.resolve(), 'Second status must not overwrite the first status'
    assert job not in (FIRST_STATUS.resolve(), status), 'Job/status paths must differ'
    first_cohort = Path(first['cohort']).resolve() if first.get('cohort') else None
    if first_cohort:
        assert cohort != first_cohort, 'Use a distinct actual second candidate cohort'
        assert (cohort/'submission').resolve() != (first_cohort/'submission').resolve(), 'Never share first upload intent/receipt area'
    assert not (cohort/'submission').is_symlink(), 'Second submission area must be local to its new cohort'
    prior = watch.read(status, {})
    assert not prior.get('cohort') or Path(prior['cohort']).resolve() == cohort, 'Existing second status belongs to a different cohort'
    protected = {}
    for path in [FIRST_STATUS, Path(first['receipt']) if first.get('receipt') else None,
                 first_cohort/'submission/upload-intent.json' if first_cohort else None]:
        if path is not None and path.is_file():
            protected[str(path)] = watch.digest(path)
    return protected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--status', type=Path, default=ROOT/'notes/track3-second-submission-status.json')
    parser.add_argument('--job', type=Path, default=ROOT/'notes/track3-second-submission-job.json')
    parser.add_argument('--profile', default='second')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--submit-on-ready', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=86400)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    cohort, status, job = (path.resolve() for path in (args.cohort, args.status, args.job))
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.profile)
    assert 1 <= args.max_seconds <= 86400 and args.poll_seconds >= 10
    protected = separate(cohort, status, job)
    command = [str(watch.PYTHON), '-u', str(Path(__file__).resolve()), '--watch', '--cohort', str(cohort),
               '--status', str(status), '--job', str(job), '--profile', args.profile,
               '--max-seconds', str(args.max_seconds), '--poll-seconds', str(args.poll_seconds)]
    if args.submit_on_ready:
        command.append('--submit-on-ready')
    session = 'v2d-t3-second-submission-'+args.profile
    log = ROOT/'logs'/('track3-second-submission-'+args.profile+'.log')
    exitfile = log.with_suffix('.exit')
    plan = dict(schema_version=1, updated_utc=watch.now(), state='watching' if args.watch else 'prepared_only', cohort=str(cohort),
                status=str(status), session=session, log=str(log), exit_file=str(exitfile),
                intent=str(cohort/'submission/upload-intent.json'), receipt='submission/attempt_<artifact_and_commit_fingerprint>/kaggle-receipt.json',
                argv=command, submit_on_ready=args.submit_on_ready, uploads_performed_by_preparation=0,
                wrapper_sha256=watch.digest(Path(__file__)), coordinator_sha256=watch.digest(Path(watch.__file__)),
                protected_first_files_sha256=protected,
                recovery='Same command and same status/cohort; existing intent polls exact receipt IDs. Ambiguous upload requires review, never automatic reupload.',
                description_note='Reused upstream coordinator retains its historical first-baseline message; use cohort/job/proof and receipt IDs to identify #2.')
    job.parent.mkdir(parents=True, exist_ok=True)
    if args.launch:
        exists = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0
        assert not exists, 'Second submission session already active; recover it instead'
        log.parent.mkdir(parents=True, exist_ok=True)
        watch.save(job, plan)
        shell = shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1; rc=$?; printf "%s\\n" "$rc" > '+shlex.quote(str(exitfile))
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
        plan.update(state='launched', updated_utc=watch.now())
        watch.save(job, plan)
        print(json.dumps(dict(state=plan['state'], session=session, job=str(job), status=str(status))))
        return 0
    watch.save(job, plan)
    # --launch is handled above so the child executes this wrapper, preserving
    # the private STATUS override. Stage workers need only their own descriptors.
    watch.STATUS = status
    sys.argv = [str(Path(watch.__file__)), '--cohort', str(cohort), '--profile', args.profile,
                '--max-seconds', str(args.max_seconds), '--poll-seconds', str(args.poll_seconds)]
    sys.argv.append('--watch' if args.watch else '--inspect')
    if args.submit_on_ready:
        sys.argv.append('--submit-on-ready')
    try:
        code = watch.main()
    except Exception as error:
        value = watch.read(status, {})
        value.update(state='second_coordinator_error', updated_utc=watch.now(), error_type=type(error).__name__,
                     attention_reason='Preserve intent and receipt; inspect persistent logs. Do not repeat uploads.')
        watch.save(status, value)
        raise
    if not args.watch:
        assert all(watch.digest(Path(path)) == digest for path,digest in protected.items()), 'First artifacts changed during local preparation'
    return code


if __name__ == '__main__':
    raise SystemExit(main())
