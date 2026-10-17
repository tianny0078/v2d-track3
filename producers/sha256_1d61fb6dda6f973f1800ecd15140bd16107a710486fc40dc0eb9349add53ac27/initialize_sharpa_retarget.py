"""Position-only Sharpa IK initializer using the official 22-joint kinematics.

Official site weights/MANO ordering and default morphology scale are reused.
This bounded numerical initializer does not reproduce PINK orientation tasks,
does not optimize contact wrench, and does not establish policy success.
"""
from datetime import datetime,timezone
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'sharpa_keypoint_initializer_v2'
SOURCE=BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz'
ASSETS=ROOT/'vendor/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave'
# Taken from NVIDIA's SHARPA_TO_MANO_MAPPING; orientation tasks are omitted here.
SITES=['hand_wrist_site','thumb_MCP_VL_site','thumb_tip_site','index_MP_site','index_tip_site','middle_MP_site','middle_tip_site','ring_MP_site','ring_tip_site','pinky_MP_site','pinky_tip_site']
JOINTS=np.array([0,2,4,5,8,9,12,13,16,17,20]);WEIGHTS=np.sqrt([.2,.1,1,.1,1,.1,1,.1,1,.1,.5])[:,None]
MORPHOLOGY_SCALE=1.2

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def model(side):
    tree=ET.parse(ASSETS/f'{side}_sharpawave.xml');root=tree.getroot()
    # Own FK-only derivative. All official joint transforms/ranges/sites/inertials
    # retained; remove mesh geometry so no physics or collision claim is made.
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {'asset','geom','actuator','contact'}:parent.remove(child)
    root.find('compiler').attrib.pop('meshdir',None)
    path=OUT/f'{side}_kinematics_only.xml';tree.write(path,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,side+'_'+site) for site in SITES]);assert (ids>=0).all()
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)]
    assert m.nq==m.nv==22 and m.jnt_limited.all()
    return m,d,ids,names

def forward(m,d,ids,q):
    d.qpos[:]=q;mujoco.mj_forward(m,d);return d.site_xpos[ids].copy()

def align(a,b):
    u,_,v=np.linalg.svd((a-a[0]).T@(b-b[0]));flip=np.eye(3);flip[-1,-1]=np.sign(np.linalg.det(v.T@u.T));return v.T@flip@u.T

def solve(m,d,ids,joints,previous):
    # A zero configuration at many lower limits makes TRF's initial trust region
    # tiny, causing a false ftol success after two evaluations. Use an interior
    # seed and physically scaled variables, retaining the first run as evidence.
    neutral=np.clip(np.full(22,.2),m.jnt_range[:,0]+.01,m.jnt_range[:,1]-.01)
    q0=np.clip(previous if previous is not None else neutral,m.jnt_range[:,0]+1e-5,m.jnt_range[:,1]-1e-5)
    target=joints[0]+(joints[JOINTS]-joints[0])*MORPHOLOGY_SCALE;local=forward(m,d,ids,q0)
    palm=np.array([0,3,5,7,9]);R0=align(local[palm],target[palm]);t0=target[0]-R0@local[0]
    x0=np.r_[np.zeros(6),q0];lower=np.r_[[-.04]*3,[-1.2]*3,m.jnt_range[:,0]+1e-8];upper=np.r_[[.04]*3,[1.2]*3,m.jnt_range[:,1]-1e-8]
    def unpack(x):return t0+x[:3],Rotation.from_rotvec(x[3:6]).as_matrix()@R0,x[6:]
    def residual(x):
        t,R,q=unpack(x);points=forward(m,d,ids,q)@R.T+t
        regularizers=[x[:3]/.015*.15,x[3:6]/.5*.04]
        if previous is not None:regularizers.append((q-previous)*.025)
        return np.concatenate([((points-target)*WEIGHTS/.01).ravel(),*regularizers])
    before=residual(x0);opt=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',f_scale=1.,max_nfev=65,ftol=1e-5,xtol=1e-6,gtol=1e-5,x_scale=np.r_[[.01]*3,[.3]*3,[.4]*22])
    t,R,q=unpack(opt.x);points=forward(m,d,ids,q)@R.T+t;errors=np.linalg.norm(points-target,axis=1)
    return {'qpos':q,'wrist_pose':np.r_[t,Rotation.from_matrix(R).as_quat(scalar_first=True)],'sites':points,
            'row':{'position_errors_m':errors.tolist(),'fingertip_error_m':errors[[2,4,6,8,10]].tolist(),
                   'wrist_displacement_from_human_m':float(np.linalg.norm(t-joints[0])),'optimizer_success':bool(opt.success),
                   'optimizer_evaluations':int(opt.nfev),'objective_before':float(np.linalg.norm(before)),
                   'objective_after':float(np.linalg.norm(residual(opt.x))),'joint_range_violations':int(((q<m.jnt_range[:,0])|(q>m.jnt_range[:,1])).sum())}}

def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic();z=np.load(SOURCE);present=z['present'].astype(bool);joints=z['joints_world']
    cache_key=hashlib.sha256(SOURCE.read_bytes()+Path(__file__).read_bytes()+b''.join((ASSETS/f'{s}_sharpawave.xml').read_bytes() for s in ['left','right'])).hexdigest()
    models=[model(s) for s in ['left','right']];q=np.full((137,2,22),np.nan);wrist=np.full((137,2,7),np.nan);sites=np.full((137,2,11,3),np.nan);done=np.zeros((137,2),bool);rows=[]
    cp=OUT/'checkpoint.npz';rp=OUT/'report.json'
    if cp.exists() and rp.exists() and json.loads(rp.read_text()).get('source_hash')==cache_key:
        old=np.load(cp);q=old['qpos'];wrist=old['wrist_pose'];sites=old['sites_world'];done=old['done'];rows=json.loads(rp.read_text()).get('frames',[])
    report={'state':'running','source_hash':cache_key,'started_utc':datetime.now(timezone.utc).isoformat(),'frames':rows,'morphology_scale':MORPHOLOGY_SCALE,
            'source_hand_candidate':'alpha_050 (not fully contact-validated)','hand_order':['left','right'],'site_suffixes':SITES,'mano_joint_indices':JOINTS.tolist(),
            'joint_names':{s:models[i][3] for i,s in enumerate(['left','right'])},'official_source_commit':'193382a184308f6800331f3db829f76dc4294cfe',
            'contact_optimized':False,'physics_rollout':False,'automatic_promotion':False,
            'limitations':['Position-only SciPy IK with official kinematics/weights; not exact official PINK orientation-task retarget.',
                           '1.2 human-to-robot morphology scale is an official default hyperparameter, not measured camera scale.',
                           'Source hands still have depth/visibility uncertainty. Missing human hands remain NaN and present=false.',
                           'FK-only derivative has no collision geometry; no contact, dynamics, policy or task success is implied.']}
    for f in range(137):
        for hi,side in enumerate(['left','right']):
            if not present[f,hi] or done[f,hi]:continue
            m,d,ids,_=models[hi];previous=q[f-1,hi] if f>0 and done[f-1,hi] else None
            result=solve(m,d,ids,joints[f,hi],previous);q[f,hi]=result['qpos'];wrist[f,hi]=result['wrist_pose'];sites[f,hi]=result['sites'];done[f,hi]=True
            rows.append({'frame':f,'side':side,**result['row']})
        np.savez_compressed(cp,qpos=q,wrist_pose=wrist,sites_world=sites,present=present,done=done,frame_index=np.arange(137))
        report.update(frames=rows,completed_hand_instances=int(done.sum()),updated_utc=datetime.now(timezone.utc).isoformat());save(rp,report)
        if f%10==0:print('frame',f,'completed',int(done.sum()),'elapsed',time.monotonic()-start,flush=True)
        if time.monotonic()-start>540:print('Budget reached; checkpoint retained',flush=True);return
    errors=np.array([r['fingertip_error_m'] for r in rows]).ravel();dq=[]
    for hi in range(2):
        consecutive=done[1:,hi]&done[:-1,hi];dq.extend(np.max(abs(np.diff(q[:,hi],axis=0)[consecutive]),axis=1).tolist())
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                  fingertip_error_p50_p95_m=np.quantile(errors,[.5,.95]).tolist(),fingertip_error_worst_m=float(errors.max()),
                  consecutive_joint_step_max_rad=float(max(dq)),joint_range_violations=sum(r['joint_range_violations'] for r in rows),
                  reconstructed_missing_hand_instances=int((done&~present).sum()),initializer_ready=bool(np.array_equal(done,present)),contact_ready=False)
    save(rp,report);print('Completed initializer',report['completed_hand_instances'],report['fingertip_error_p50_p95_m'],flush=True)

if __name__=='__main__':main()
