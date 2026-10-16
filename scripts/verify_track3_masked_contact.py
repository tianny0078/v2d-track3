"""Bounded CPU verification of project Warp kernels; no policy training."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import warp as wp
import track3_masked_contact as masked
from flash_chord.objectives import contact as official
from track3_wrench_adapter import masked_contact_terms
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track3/episode_000011/masked_contact_kernels'
NAMES=['contact_wrench_support_objective','missed_contact_penalty','unintended_contact_penalty','force_closure_objective']
def evaluate(ref,cur,known=None):
    W,H,B,K=ref.shape
    args=[wp.array(ref.ravel(),dtype=wp.float32,device='cpu'),wp.array(cur.ravel(),dtype=wp.float32,device='cpu')]
    if known is not None:args.append(wp.array(known.astype(np.int32).ravel(),dtype=wp.int32,device='cpu'))
    outputs=[]
    for name,extra in zip(NAMES,[[.1,1.],[],[],[.01]]):
        out=wp.zeros(W,dtype=wp.float32,device='cpu')
        wp.launch(getattr(masked if known is not None else official,name),dim=W,inputs=[*args,H,B,K,*extra],outputs=[out],device='cpu')
        outputs.append(out.numpy().copy())
    return np.stack(outputs,-1)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);r=rng.uniform(-.2,2,(9,2,3,32)).astype(np.float32);c=rng.uniform(-.2,2,r.shape).astype(np.float32)
    r[:,0,0]=0;r[0]=0;c[1]=0
    k=np.ones(r.shape[:-1],bool);baseline=evaluate(r,c);allknown=evaluate(r,c,k)
    error=float(abs(baseline-allknown).max());assert error<2e-6,error
    k=rng.random(k.shape)>.45;k[0]=False
    rn=np.where(k[...,None],r,np.nan);cn=np.where(k[...,None],c,np.nan)
    actual=evaluate(rn,cn,k);expected=masked_contact_terms(rn,cn,k)['terms']
    mixederror=float(abs(actual-expected).max());assert mixederror<2e-6,mixederror
    assert np.isfinite(actual).all() and (actual[0]==0).all()
    # Known inactive body plus unknown active body: denominator must be one.
    rr=np.zeros((1,1,2,8),np.float32);cc=np.ones_like(rr);kk=np.array([[[True,False]]])
    special=evaluate(rr,cc,kk);assert abs(special[0,2]-2)<1e-6
    # Same clamped per-world timestep semantics as official objective gather,
    # including independently reset worlds and end-of-reference clamping.
    refk=rng.integers(0,2,(7,2,3),dtype=np.int32);times=np.array([-1,0,2,6,99],np.int32)
    kd=wp.array(refk.ravel(),dtype=wp.int32,device='cpu');td=wp.array(times,dtype=wp.int32,device='cpu');out=wp.zeros(30,dtype=wp.int32,device='cpu')
    for ts in [times,np.array([0,4,0,6,3],np.int32)]:
        td.assign(ts);wp.launch(masked.gather_known,dim=(5,6),inputs=[td,kd,7,6],outputs=[out],device='cpu')
        assert np.array_equal(out.numpy().reshape(5,2,3),refk[np.clip(ts,0,6)])
    # Replay genuine existing physical contact support snapshots without rerunning physics.
    z=np.load(ROOT/'outputs/track3/episode_000011/wrench_origin_semantics/physical_supports.npz')
    physical=z['runtime_COM'].reshape(3,2,2,-1).astype(np.float32);pk=np.ones((3,2,2),bool);pk[:,1]=False
    physical_terms=evaluate(physical,physical,pk);np.savez_compressed(OUT/'checkpoint.npz',reference=rn,current=cn,known=k,actual=actual,expected=expected,physical_terms=physical_terms)
    report=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pass_all=True,all_known_max_abs_error=error,mixed_unknown_max_abs_error=mixederror,all_unknown_finite_zero=True,unknown_nan_payload_ignored=True,unintended_known_denominator_value=float(special[0,2]),batched_reset_gather_pass=True,physical_snapshot_frames=3,new_physics_steps=0,production_integrated=False,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/track3_masked_contact.py']},remaining=['Wire project Objective subclass into actual training entrypoint','COM reference view and runtime slot transform using scene body COM and matching radius','Gather known mask using objective timestep on every reward/reset; replace four objective buffers before composition','Run real small-world step/reward integration test; no production claim before this passes'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
