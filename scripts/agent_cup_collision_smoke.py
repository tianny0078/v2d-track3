"""Bounded hollow-cup compound-convex collision/dynamics experiment.

Each body has 64 narrow convex wall wedges plus a convex bottom frustum. No
whole-cup convex hull is used. Tests do not replay or validate recovered motion.
"""
from __future__ import annotations
import json,time,hashlib,traceback,os
from pathlib import Path
import numpy as np
import trimesh
import mujoco
from scipy.spatial.transform import Rotation
from agent_cup_reconstruct import ROOT,BASE,NAMES,save,now

OUT=BASE/'agent_cup_collision_smoke'
SOURCE=BASE/'agent_cup_reconstruction'
SEGMENTS=64;BOTTOM=.003;MASS=.12;FRICTION=.5;MARGIN=.0001;DT=.001;BUDGET=540

def pieces_for(s,t):
    h=s['height_m'];r=s['bottom_outer_radius_m'];k=s['taper_dr_dz'];pieces=[]
    for i in range(SEGMENTS):
        a=np.array([i,i+1])*2*np.pi/SEGMENTS;vertices=[]
        for z in [BOTTOM,h]:
            for rad in [r+k*z-t,r+k*z]:
                for aa in a:vertices.append([rad*np.cos(aa),rad*np.sin(aa),z])
        mesh=trimesh.convex.convex_hull(np.array(vertices));mesh.fix_normals();pieces.append(mesh)
    a=np.arange(SEGMENTS)*2*np.pi/SEGMENTS
    vertices=np.concatenate([np.column_stack(((r+k*z)*np.cos(a),(r+k*z)*np.sin(a),np.full(SEGMENTS,z))) for z in [0,BOTTOM]])
    bottom=trimesh.convex.convex_hull(vertices);bottom.fix_normals();pieces.append(bottom)
    return pieces

def props(pieces):
    mesh=trimesh.util.concatenate(pieces);p=mesh.mass_properties
    return {'volume_m3':float(p.volume),'com_m':p.center_mass.tolist(),'inertia_kg_m2':(p.inertia*MASS/p.volume).tolist(),'mass_kg':MASS,'density_implied_kg_m3':float(MASS/p.volume)}

def fmt(a):return ' '.join(f'{float(x):.10g}' for x in np.asarray(a).ravel())
def xml_for(assets,properties,mu=.5,fixed_blue=False,positions=None):
    positions=positions or [[-.12,0,.015],[.12,0,.015]]
    xml=['<mujoco model="hollow_cup_compound"><compiler angle="radian"/><option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" iterations="80" tolerance="1e-9" cone="elliptic"/><size nconmax="4096" njmax="12000"/><default><geom contype="1" conaffinity="1" condim="4" margin="0.0001" solref="0.005 1" solimp="0.9 0.95 0.001" friction="'+f'{mu} 0.002 0.0001'+'"/></default><asset>']
    for oi,parts in enumerate(assets):
        for pi,p in enumerate(parts):xml.append(f'<mesh name="m{oi}_{pi}" vertex="{fmt(p.vertices)}" face="{fmt(p.faces.astype(int))}"/>')
    xml.append('</asset><worldbody><geom name="table" type="plane" size="1 1 .05"/>')
    for oi,name in enumerate(NAMES):
        p=properties[oi];inertia=np.asarray(p['inertia_kg_m2']);com=p['com_m']
        xml.append(f'<body name="{name}" pos="{fmt(positions[oi])}">')
        if not(oi==0 and fixed_blue):xml.append(f'<freejoint name="{name}_free"/>')
        xml.append(f'<inertial pos="{fmt(com)}" mass="{MASS}" fullinertia="{fmt([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]])}"/>')
        for pi,_ in enumerate(assets[oi]):xml.append(f'<geom name="{name}_part_{pi}" type="mesh" mesh="m{oi}_{pi}" density="0"/>')
        xml.append('</body>')
    xml.append('</worldbody></mujoco>');return ''.join(xml)

def body_id(model,name):return mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
def set_body_pose(model,data,name,p,quat=(1,0,0,0)):
    bi=body_id(model,name);ji=int(model.body_jntadr[bi]);q=int(model.jnt_qposadr[ji]);data.qpos[q:q+3]=p;data.qpos[q+3:q+7]=quat

def contact_metrics(model,data):
    blue=body_id(model,NAMES[0]);beige=body_id(model,NAMES[1]);dist=[];table=[]
    for c in data.contact[:data.ncon]:
        bodies={int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])}
        if bodies=={blue,beige}:dist.append(float(c.dist))
        elif 0 in bodies and (blue in bodies or beige in bodies):table.append(float(c.dist))
    return {'cup_contacts':len(dist),'cup_min_contact_distance_m':min(dist) if dist else None,'cup_penetrating_contacts_over_10um':sum(d<-.00001 for d in dist),'table_contacts':len(table),'table_min_contact_distance_m':min(table) if table else None,'total_contacts':int(data.ncon)}

def static_queries(model,shapes,deadline):
    d=mujoco.MjData(model);rows=[];h=shapes[0]['height_m']
    for insertion in np.arange(-.01,h+.0001,.002):
        if time.monotonic()>deadline:break
        mujoco.mj_resetData(model,d);set_body_pose(model,d,NAMES[0],[0,0,0]);set_body_pose(model,d,NAMES[1],[0,0,h-insertion]);mujoco.mj_forward(model,d)
        rows.append({'insertion_m':float(insertion),**contact_metrics(model,d)})
    clear=[x['insertion_m'] for x in rows if x['cup_penetrating_contacts_over_10um']==0];maximum=max(clear) if clear else None
    # All rows retained: a max alone could hide an erroneous re-entry interval.
    challenges=[];safe=.08
    for dx in [0,.003,.008,.015,.025]:
        for angle in [0,5,12]:
            mujoco.mj_resetData(model,d);set_body_pose(model,d,NAMES[0],[0,0,0]);q=Rotation.from_euler('x',angle,degrees=True).as_quat();set_body_pose(model,d,NAMES[1],[dx,0,h-safe],np.r_[q[3],q[:3]]);mujoco.mj_forward(model,d)
            challenges.append({'insertion_m':safe,'offset_x_m':dx,'tilt_deg':angle,**contact_metrics(model,d)})
    return {'coaxial_sweep':rows,'maximum_sampled_nonpenetrating_insertion_m':maximum,'side_tilt_queries':challenges,'mouth_allows_80mm_coaxial_insertion':bool(challenges[0]['cup_penetrating_contacts_over_10um']==0),'lateral_conflict_detected':any(x['offset_x_m']>=.015 and x['cup_penetrating_contacts_over_10um']>0 for x in challenges),'tilt_conflict_detected':any(x['tilt_deg']==12 and x['cup_penetrating_contacts_over_10um']>0 for x in challenges)}

def simulate(model,seconds,deadline,tag):
    d=mujoco.MjData(model);mujoco.mj_forward(model,d);trace=[];ids=[body_id(model,n) for n in NAMES];warning0=np.array([w.number for w in d.warning]);maxpenetration=0.;maxcontacts=0
    for step in range(int(seconds/DT)):
        if step%100==0 and time.monotonic()>deadline:break
        mujoco.mj_step(model,d)
        if step%20==0:
            cm=contact_metrics(model,d);ds=[x for x in [cm['cup_min_contact_distance_m'],cm['table_min_contact_distance_m']] if x is not None];maxpenetration=max(maxpenetration,max([-x for x in ds],default=0));maxcontacts=max(maxcontacts,d.ncon)
            trace.append({'time':float(d.time),'body_pos':d.xpos[ids].tolist(),'body_quat_wxyz':d.xquat[ids].tolist(),'qvel_norm':float(np.linalg.norm(d.qvel)),**cm})
    warnings=[{'index':i,'count':int(w.number-warning0[i])} for i,w in enumerate(d.warning) if w.number>warning0[i]]
    final=trace[-1];tail=trace[-min(25,len(trace)):];pos=np.array([r['body_pos'] for r in tail]);drift=np.linalg.norm(pos[-1]-pos[0],axis=1)
    axes=[Rotation.from_quat(np.r_[q[1:],q[0]]).apply([0,0,1]) for q in final['body_quat_wxyz']];tilts=[float(np.rad2deg(np.arccos(np.clip(a[2],-1,1)))) for a in axes]
    result={'tag':tag,'simulation_seconds':float(d.time),'finite_state':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),'warnings':warnings,'max_sampled_penetration_m':float(maxpenetration),'max_sampled_contact_count':int(maxcontacts),'final':final,'last_0_5s_position_drift_m':drift.tolist(),'final_tilt_deg':tilts,'settled_gate':bool(not warnings and max(drift)<.002 and max(tilts)<5 and maxpenetration<.003),'trace':trace}
    return result

def newton_smoke(assets,properties,shapes,dest,deadline):
    import warp as wp
    import newton
    from newton.solvers import SolverMuJoCo
    wp.init();builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder)
    cfg=newton.ModelBuilder.ShapeConfig(density=0.,mu=FRICTION,margin=MARGIN,mu_torsional=.002,mu_rolling=.0001)
    builder.add_ground_plane(cfg=cfg,label='table')
    for oi,name in enumerate(NAMES):
        p=properties[oi];I=np.asarray(p['inertia_kg_m2'],dtype=np.float32)
        b=builder.add_body(xform=wp.transform(wp.vec3((oi-.5)*.24,0,.015),wp.quat_identity()),mass=MASS,com=wp.vec3(*p['com_m']),inertia=wp.mat33(*I.ravel()),lock_inertia=True,label=name)
        # Newton 1.2 add_body already creates the free joint and articulation.
        for pi,mesh in enumerate(assets[oi]):
            nm=newton.Mesh(np.asarray(mesh.vertices,dtype=np.float32),np.asarray(mesh.faces,dtype=np.int32).ravel(),compute_inertia=False,is_solid=True)
            builder.add_shape_mesh(b,mesh=nm,cfg=cfg,label=f'{name}_part_{pi}')
    model=builder.finalize(device='cpu');solver=SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,update_data_interval=0,nconmax=4096,njmax=12000,iterations=80,integrator='implicitfast',cone='elliptic',save_to_mjcf=str(dest/'newton_export.xml'))
    a=model.state();b=model.state();control=model.control();contacts=model.contacts();trace=[]
    for step in range(1000):
        if step%50==0 and time.monotonic()>deadline:break
        a.clear_forces();solver.step(a,b,control,contacts,DT);a,b=b,a
        if step%20==0:trace.append({'time':(step+1)*DT,'newton_body_q':a.body_q.numpy().tolist(),'newton_body_qd':a.body_qd.numpy().tolist(),'native_contact_count':int(solver.mj_data.ncon)})
    # Exercise actual Newton-exported geometry through its native MuJoCo model.
    q=static_queries(solver.mj_model,shapes,deadline)
    arr=a.body_q.numpy();drift=np.linalg.norm(np.array(trace[-1]['newton_body_q'])[:,:3]-np.array(trace[-min(25,len(trace))]['newton_body_q'])[:,:3],axis=1)
    return {'state':'completed','tested':True,'backend':'Newton 1.2.0 SolverMuJoCo(use_mujoco_cpu=True), native MuJoCo contacts','actual_newton_solver_steps':step+1,'finite_state':bool(np.isfinite(arr).all()),'last_0_5s_drift_m':drift.tolist(),'body_mass_kg':model.body_mass.numpy().tolist(),'body_com_m':model.body_com.numpy().tolist(),'rest_gate':bool(np.isfinite(arr).all() and max(drift)<.002 and np.all(arr[:,2]>-.003)),'trace':trace,'native_collision_queries':q,'newton_standard_collision_pipeline_tested':False,'gpu_backend_tested':False}

def resume_newton():
    """Repair backend integration without regenerating or rerunning native assets."""
    report=json.loads((OUT/'report.json').read_text())
    source_key=str((SOURCE/'report.json').relative_to(ROOT))
    if hashlib.sha256((SOURCE/'report.json').read_bytes()).hexdigest()!=report['source_sha256'][source_key]:
        raise RuntimeError('Source geometry changed: preserve this experiment and use a new output directory.')
    if report.get('newton',{}).get('state')=='completed':
        print('Newton backend already completed; retained results',flush=True);return
    original=OUT/'report_before_newton_repair.json'
    if not original.exists():save(original,report)
    start=time.monotonic();report.update(state='running_newton_repair',updated_utc=now());save(OUT/'report.json',report)
    source=json.loads((SOURCE/'report.json').read_text());shapes=[source['objects'][n]['shape'] for n in NAMES]
    assets=[pieces_for(s,.003) for s in shapes];properties=[props(p) for p in assets]
    try:
        result=newton_smoke(assets,properties,shapes,OUT/'wall_3000um',start+BUDGET)
        save(OUT/'newton_backend.json',result)
        report['newton']={k:v for k,v in result.items() if k not in ['trace','native_collision_queries']}
        report['newton']['collision_query_summary']={k:v for k,v in result['native_collision_queries'].items() if k not in ['coaxial_sweep','side_tilt_queries']}
    except Exception:
        report['newton']={'state':'failed','tested':False,'error':traceback.format_exc()}
        save(OUT/f'newton_backend_repair_failure_{int(time.time())}.json',report['newton'])
    report.update(state='completed' if report['newton']['state']=='completed' else 'completed_partial',completed_utc=now(),newton_repair_elapsed_seconds=time.monotonic()-start,newton_repair_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_stages_reused_without_rerun=True)
    save(OUT/'report.json',report);print('NEWTON REPAIR',json.dumps(report['newton']),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();deadline=start+BUDGET;source=json.loads((SOURCE/'report.json').read_text());shapes=[source['objects'][n]['shape'] for n in NAMES]
    report={'state':'running','started_utc':now(),'budget_seconds':BUDGET,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'report.json',Path(__file__)]},'method':'64 convex annular frustum wedges plus separate bottom frustum per cup; never a whole-cup convex hull','assumptions':{'wall_thickness_candidates_m':[.0015,.003,.005],'bottom_thickness_m':BOTTOM,'mass_per_cup_kg':MASS,'mass_measured':False,'friction':FRICTION,'friction_measured':False,'contact_margin_m':MARGIN,'time_step_s':DT,'gravity_m_s2':9.81,'uniform_density':True,'geometry':'v3 inferred outer frusta; unmeasured inner walls/base and excluded markers'},'training':False,'trajectory_replay':False,'versions':{'mujoco':mujoco.__version__,'trimesh':trimesh.__version__},'candidates':{},'limitations':['Asset/backend smoke only; reconstructed trajectories remain failed.','Convex segmentation approximates circular walls; piece seams and solver parameters require task-level testing.','No true mass, friction, wall thickness or cup-bottom measurement.','Newton standard collision pipeline and GPU backend are distinct from the tested native MuJoCo collision path.']}
    cache_key=hashlib.sha256(json.dumps(report['source_sha256'],sort_keys=True).encode()).hexdigest()
    save(OUT/'report.json',report);nominal=None
    for t in [.0015,.003,.005]:
        if time.monotonic()>deadline:break
        tag=f'wall_{int(t*1e6)}um';folder=OUT/tag;folder.mkdir(exist_ok=True);assets=[];properties=[];quality=[]
        for oi,name in enumerate(NAMES):
            parts=pieces_for(shapes[oi],t);assets.append(parts);properties.append(props(parts));dest=folder/name;dest.mkdir(exist_ok=True)
            for pi,m in enumerate(parts):m.export(dest/f'part_{pi:03d}.obj')
            trimesh.util.concatenate(parts).export(dest/'compound_visual.ply')
            quality.append({'object':name,'convex_parts':len(parts),'all_parts_convex':bool(all(m.is_convex for m in parts)),'all_parts_watertight':bool(all(m.is_watertight for m in parts)),'all_parts_positive_volume':bool(all(m.volume>0 for m in parts)),'inner_polygon_radial_shrink_bound_m':float((shapes[oi]['bottom_outer_radius_m']+shapes[oi]['taper_dr_dz']*shapes[oi]['height_m']-t)*(1-np.cos(np.pi/SEGMENTS))),'properties':properties[-1]})
        cached=folder/'stage_checkpoint.json'
        if cached.exists() and json.loads(cached.read_text()).get('cache_key')==cache_key:
            report['candidates'][tag]=json.loads(cached.read_text())['result'];save(OUT/'report.json',report)
            if t==.003:nominal=(assets,properties,folder)
            print(tag,'reused completed collision/dynamics stage',flush=True);continue
        xml=xml_for(assets,properties);(folder/'scene.xml').write_text(xml);model=mujoco.MjModel.from_xml_string(xml)
        static=static_queries(model,shapes,deadline);save(folder/'static_queries.json',static)
        rest=simulate(model,2.,deadline,'two_separate_free_cups_drop_15mm');save(folder/'table_rest.json',rest)
        nestedxml=xml_for(assets,properties,fixed_blue=True,positions=[[0,0,0],[0,0,shapes[0]['height_m']-.08]]);(folder/'nesting_scene.xml').write_text(nestedxml)
        nested=simulate(mujoco.MjModel.from_xml_string(nestedxml),2.,deadline,'beige_free_drop_into_fixed_blue_from_80mm_insertion');save(folder/'nesting_drop.json',nested)
        result={'wall_thickness_m':t,'geometry_quality':quality,'static':{k:v for k,v in static.items() if k not in ['coaxial_sweep','side_tilt_queries']},'table_rest':{k:v for k,v in rest.items() if k!='trace'},'nesting_drop':{k:v for k,v in nested.items() if k!='trace'},'artifacts':str(folder.relative_to(ROOT))};report['candidates'][tag]=result;save(cached,{'cache_key':cache_key,'result':result});save(OUT/'report.json',report)
        print(tag,'max_insert',static['maximum_sampled_nonpenetrating_insertion_m'],'rest',rest['settled_gate'],'nest',nested['settled_gate'],'time',time.monotonic()-start,flush=True)
        if t==.003:nominal=(assets,properties,folder)
    if nominal and time.monotonic()<deadline:
        assets,properties,folder=nominal
        try:
            newton_result=newton_smoke(assets,properties,shapes,folder,deadline);save(OUT/'newton_backend.json',newton_result);report['newton']={k:v for k,v in newton_result.items() if k not in ['trace','native_collision_queries']};report['newton']['collision_query_summary']={k:v for k,v in newton_result['native_collision_queries'].items() if k not in ['coaxial_sweep','side_tilt_queries']}
        except Exception:report['newton']={'state':'failed','error':traceback.format_exc(),'tested':False};save(OUT/'newton_backend_failure.json',report['newton'])
    else:report['newton']={'state':'not_started_budget','tested':False}
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,geometry_can_represent_nesting=bool(all(c['static']['mouth_allows_80mm_coaxial_insertion'] and c['static']['lateral_conflict_detected'] and c['static']['tilt_conflict_detected'] for c in report['candidates'].values())),reconstruction_reference_pass=False)
    save(OUT/'report.json',report);print('COMPLETED',report['elapsed_seconds'],'nesting_representation',report['geometry_can_represent_nesting'],'newton',report['newton'],flush=True)

if __name__=='__main__':
    try:
        prior=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
        if len(prior.get('candidates',{}))==3:resume_newton()
        else:main()
    except Exception:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error=traceback.format_exc(),updated_utc=now());save(OUT/'report.json',r)
        raise
