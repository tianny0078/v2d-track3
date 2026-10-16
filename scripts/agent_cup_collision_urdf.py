"""Export nominal compound URDFs and check Newton import without replaying dynamics."""
import json,os,time
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo
from agent_cup_collision_smoke import OUT,SOURCE,NAMES,save,now,FRICTION,MARGIN

def main():
    start=time.monotonic();report=json.loads((OUT/'report.json').read_text());quality=report['candidates']['wall_3000um']['geometry_quality'];rows=[]
    for name,q in zip(NAMES,quality):
        folder=OUT/'wall_3000um'/name;p=q['properties'];I=np.asarray(p['inertia_kg_m2']);com=p['com_m'];visual=SOURCE/name/'wall_03mm_3000um.obj'
        xyz=' '.join(f'{x:.12g}' for x in com)
        xml=[f'<robot name="{name}"><link name="{name}"><inertial><origin xyz="{xyz}" rpy="0 0 0"/><mass value="{p["mass_kg"]}"/><inertia ixx="{I[0,0]:.12g}" iyy="{I[1,1]:.12g}" izz="{I[2,2]:.12g}" ixy="{I[0,1]:.12g}" ixz="{I[0,2]:.12g}" iyz="{I[1,2]:.12g}"/></inertial>',f'<visual><geometry><mesh filename="{os.path.relpath(visual,folder)}" scale="1 1 1"/></geometry></visual>']
        for i in range(65):xml.append(f'<collision name="part_{i:03d}"><geometry><mesh filename="part_{i:03d}.obj" scale="1 1 1"/></geometry></collision>')
        xml.append('</link></robot>');path=folder/f'{name}.urdf';path.write_text('\n'.join(xml));rows.append({'object':name,'urdf_path':str(path),'collision_parts':65,'properties':p})
    wp.init();builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu=FRICTION;builder.default_shape_cfg.margin=MARGIN
    for i,row in enumerate(rows):builder.add_urdf(row['urdf_path'],floating=True,xform=wp.transform(wp.vec3((i-.5)*.24,0,.015),wp.quat_identity()),parse_visuals_as_colliders=False,ignore_inertial_definitions=False,enable_self_collisions=False)
    model=builder.finalize(device='cpu');solver=SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,save_to_mjcf=str(OUT/'wall_3000um/urdf_import_export.xml'))
    expected_mass=np.array([q['properties']['mass_kg'] for q in quality]);expected_com=np.array([q['properties']['com_m'] for q in quality]);expected_inertia=np.array([q['properties']['inertia_kg_m2'] for q in quality])
    flags=model.shape_flags.numpy();collision_count=int(np.count_nonzero(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)))
    actual_mass=model.body_mass.numpy();actual_com=model.body_com.numpy();actual_inertia=model.body_inertia.numpy()
    r={'state':'completed','completed_utc':now(),'elapsed_seconds':time.monotonic()-start,'assets':rows,'canonical_frame':'Base center z=0, axis +Z, meters; link frame is NOT COM. Inertial origin is canonical COM, inertia about COM in canonical axes.','mass_friction_and_inner_geometry_measured':False,'material_assumptions':'Mass 0.12 kg, friction 0.5, 3 mm radial walls and 3 mm bottom are assumptions. URDF stores mass/inertia; configure friction in consumer.','newton_body_count':model.body_count,'newton_joint_count':model.joint_count,'newton_shape_count_including_visual':model.shape_count,'newton_collision_shape_count':collision_count,'mujoco_nbody_including_world':solver.mj_model.nbody,'mujoco_ngeom':solver.mj_model.ngeom,'actual_mass':actual_mass.tolist(),'actual_com':actual_com.tolist(),'actual_inertia':actual_inertia.tolist(),'mass_match':bool(np.allclose(actual_mass,expected_mass,rtol=1e-6)),'com_match':bool(np.allclose(actual_com,expected_com,atol=1e-7)),'inertia_match':bool(np.allclose(actual_inertia,expected_inertia,atol=1e-9)),'dynamics_rerun':False}
    r['import_compile_gate']=bool(model.body_count==2 and model.joint_count==2 and collision_count==130 and r['mass_match'] and r['com_match'] and r['inertia_match'])
    save(OUT/'urdf_import_report.json',r);report['urdf']={k:v for k,v in r.items() if k not in ['actual_inertia']};save(OUT/'report.json',report);print(json.dumps(r),flush=True)

if __name__=='__main__':main()
