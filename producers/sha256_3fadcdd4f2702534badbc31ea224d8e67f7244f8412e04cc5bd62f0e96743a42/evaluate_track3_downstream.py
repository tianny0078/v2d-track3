"""Actual policy evaluator; uses pinned official harness with one schema adapter."""
import argparse
from pathlib import Path
import json
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--episode',type=int,default=11)
    args=parser.parse_args()
    # Load the modules belonging to this checkpoint's own immutable producer.
    snapshot=Path(args.checkpoint).resolve().parent.parent/'producer-source'
    if not snapshot.is_dir():
        raise ValueError('Checkpoint producer-source directory is missing')
    sys.path.insert(0,str(snapshot))
    from track3_evaluation_harness import evaluate_checkpoint, evaluation_scalars, write_evaluation_report
    out=Path(args.output);out.mkdir(exist_ok=True)
    result=evaluate_checkpoint(args.checkpoint,world_count=1,start_frame=0,reset_mode='explicit',
        object_trajectories_output=out/f'episode_{args.episode:06d}.parquet',eval_script=Path(__file__))
    write_evaluation_report(out/'evaluation.json',result)
    print(json.dumps(evaluation_scalars(result),indent=2),flush=True)


if __name__=='__main__':main()
