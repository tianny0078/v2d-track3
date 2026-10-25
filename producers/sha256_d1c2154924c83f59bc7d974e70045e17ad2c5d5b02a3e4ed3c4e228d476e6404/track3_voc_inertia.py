"""Project-owned VOC root torque with the full rotated inertia tensor (learning_v6i+).

The vendor kernel scales the angular PD acceleration by the MEAN principal inertia. For slender bodies (wooden_spoon:
mean/min inertia 35.6) that gives the long axis an angular damping rate d_ang*35.6 = 1424 1/s, applied as an explicit
body force at sim_dt = 0.01 s (d*dt = 14 >> 2): the spoon spins up on the first step, object termination fires at
episode length ~1.2 and SAC metrics become NaN. Here torque = R I R^T (k_ang*err - d_ang*w), i.e. the configured gains are
the angular acceleration on every axis (d_ang*dt = 0.4). Linear force, gravity compensation, clamps and articulation
targets are unchanged. At voc_scale 0 (evaluation) the wrench is zero exactly as before.
"""
import warp as wp
from flash_chord.runtime import object_control as oc
from flash_chord.runtime.object_control import VOCParams
from flash_chord.utils.quat import quat_to_rotvec

@wp.kernel
def _object_control_tensor(body_q:wp.array(dtype=wp.transform),body_qd:wp.array(dtype=wp.spatial_vector),body_mass:wp.array(dtype=wp.float32),
  body_com:wp.array(dtype=wp.vec3),body_inertia:wp.array(dtype=wp.mat33),object_root_body_ids:wp.array(dtype=wp.int32),
  object_body_offsets:wp.array(dtype=wp.int32),object_body_ids:wp.array(dtype=wp.int32),target_pos:wp.array(dtype=wp.vec3),
  target_quat:wp.array(dtype=wp.quat),articulation_dof_ids:wp.array(dtype=wp.int32),articulation_target_pos:wp.array(dtype=wp.float32),
  k_lin:float,d_lin:float,k_ang:float,d_ang:float,max_force:float,max_torque:float,scale:wp.array(dtype=wp.float32),scale_count:int,
  world_count:int,object_count:int,articulation_count:int,articulations_per_world:int,bodies_per_world:int,num_joint_dof:int,
  gravity:float,mass_normalized_gains:bool,body_f:wp.array(dtype=wp.spatial_vector),joint_target_pos:wp.array(dtype=wp.float32),
  joint_target_vel:wp.array(dtype=wp.float32)):
    i=wp.tid()
    if i<object_count:
        objects_per_world=object_count//world_count;world=i//objects_per_world
        scale_index=0
        if scale_count>1:scale_index=world
        scale_value=scale[scale_index];object_id=i-world*objects_per_world;bid=object_root_body_ids[i]
        xf=body_q[bid];pos=wp.transform_get_translation(xf);quat=wp.transform_get_rotation(xf);vel=body_qd[bid]
        ang_vel=wp.spatial_bottom(vel);com_offset_w=wp.quat_rotate(quat,body_com[bid])
        lin_vel=wp.spatial_top(vel)-wp.cross(ang_vel,com_offset_w)
        mass=float(0.0)
        for body_index in range(object_body_offsets[object_id],object_body_offsets[object_id+1]):
            mass+=body_mass[world*bodies_per_world+object_body_ids[body_index]]
        force=k_lin*(target_pos[i]-pos)-d_lin*lin_vel
        torque=k_ang*quat_to_rotvec(wp.mul(target_quat[i],wp.quat_inverse(quat)))-d_ang*ang_vel
        if mass_normalized_gains:
            R=wp.quat_to_matrix(quat);force=mass*force;torque=(R*body_inertia[bid]*wp.transpose(R))*torque
        force=force+wp.vec3(0.0,0.0,gravity*mass)
        force=wp.vec3(wp.clamp(scale_value*force[0],-max_force,max_force),wp.clamp(scale_value*force[1],-max_force,max_force),wp.clamp(scale_value*force[2],-max_force,max_force))
        torque=wp.vec3(wp.clamp(scale_value*torque[0],-max_torque,max_torque),wp.clamp(scale_value*torque[1],-max_torque,max_torque),wp.clamp(scale_value*torque[2],-max_torque,max_torque))
        body_f[bid]=wp.spatial_vector(force,torque)
    if i<articulation_count:
        world=i//articulations_per_world;articulation=i-world*articulations_per_world
        dof_id=world*num_joint_dof+articulation_dof_ids[articulation]
        joint_target_pos[dof_id]=articulation_target_pos[i];joint_target_vel[dof_id]=0.0

def install():
    """Swap the vendor root-control kernel; apply_object_control looks it up as a module global at call time."""
    oc._object_control=_object_control_tensor
    return VOCParams
