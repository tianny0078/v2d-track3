"""Project-owned explicit link/COM wrench transforms and uncertainty reductions.

No vendor patch or automatic training integration. Points use meters, normals
are directions, and the official kernel alone divides moments by COM radius.
"""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from flash_chord.objectives.wrench import wrench_support,sample_wrench_basis,friction_cone_angles

def convert_origin(points,body_com_local,*,source_origin,target_origin):
    if source_origin not in ('link','com') or target_origin not in ('link','com'):raise ValueError('Explicit link/com origin required')
    p=np.asarray(points).copy();c=np.asarray(body_com_local)
    if source_origin!=target_origin:p=p-c if target_origin=='com' else p+c
    return p

def world_to_origin(points_world,directions_world,body_position,body_quaternion_xyzw,body_com_local,*,target_origin):
    R=Rotation.from_quat(body_quaternion_xyzw).as_matrix();p=np.einsum('...ji,...j->...i',R,np.asarray(points_world)-body_position);n=np.einsum('...ji,...j->...i',R,directions_world)
    return convert_origin(p,body_com_local,source_origin='link',target_origin=target_origin),n

@wp.kernel
def runtime_link_to_com(points_link:wp.array(dtype=wp.vec3),object_ids:wp.array(dtype=wp.int32),object_com:wp.array(dtype=wp.vec3),points_com:wp.array(dtype=wp.vec3)):
    i=wp.tid();points_com[i]=points_link[i]-object_com[object_ids[i]]

def supports(points,directions,radii,object_ids,*,mu=.3,num_basis=512,device='cpu'):
    """[cells,slots,3] -> [cells,basis], shared object-seeded basis with rc=1."""
    p=np.asarray(points,dtype=np.float32);n=np.asarray(directions,dtype=np.float32);cells,slots,_=p.shape
    if n.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(n).all():raise ValueError('Finite aligned support inputs required; carry unknown separately')
    basis=np.stack([sample_wrench_basis(num_basis,1.,seed=int(i)) for i in object_ids]);cos,sin=friction_cone_angles(8);out=wp.zeros(cells*num_basis,dtype=wp.float32,device=device)
    wp.launch(wrench_support,dim=cells*num_basis,inputs=[wp.array(p.reshape(-1,3),dtype=wp.vec3,device=device),wp.array(n.reshape(-1,3),dtype=wp.vec3,device=device),wp.ones(slots,dtype=wp.int32,device=device),wp.array(basis.reshape(-1,6),dtype=wp.spatial_vector,device=device),wp.array(cos,dtype=wp.float32,device=device),wp.array(sin,dtype=wp.float32,device=device),float(mu),wp.array(radii,dtype=wp.float32,device=device),slots,8,num_basis],outputs=[out],device=device)
    return out.numpy().reshape(cells,num_basis)

def transport_known_support(values,known):
    """Finite transport payload PLUS mandatory validity mask; zeros alone are unsafe."""
    v=np.asarray(values);k=np.asarray(known,bool)
    if v.shape[:-1]!=k.shape:raise ValueError('Known mask must be [world,hand,object], independent of support activity')
    if not np.isfinite(v[k]).all():raise ValueError('Known cells must be finite')
    return np.where(k[...,None],v,0.),k.copy()

def masked_contact_terms(reference,current,known,*,tolerance=.1,var=1.,min_support=.01):
    """Host semantic prototype, preserving official formulas when all cells known.

    Unknown cells are excluded from reference, current, and all denominators.
    This has NOT been wired into the environment or optimization/replay buffers.
    """
    ref,k=transport_known_support(reference,known);cur,_=transport_known_support(current,known);W,H,B,K=ref.shape;out=np.zeros((W,4));valid=np.zeros((W,4),bool)
    for w in range(W):
        matches=[];misses=[];closures=[];unintended=0.
        for h in range(H):
            match=[];miss=[];closure=[];inactive=[];unintended_count=0
            for b in range(B):
                if not k[w,h,b]:continue
                c=ref[w,h,b];u=cur[w,h,b];expected=c>1e-3;active=u>1e-3
                if expected.any():
                    loss=np.maximum((1-tolerance)*c-u,0)**2+np.maximum(u-(1+tolerance)*c,0)**2
                    match.append(float((np.exp(-loss/var)*active)[expected].mean()));miss.append(float((~active)[expected].mean()));closure.append(float((u>min_support).mean()))
                else:inactive.append(float(np.mean(np.maximum(u,0)**2)));unintended_count+=int(active.any())
            if match:matches.append(np.mean(match));misses.append(np.mean(miss));closures.append(np.mean(closure))
            if k[w,h].any():unintended+=unintended_count/k[w,h].sum()+(np.mean(inactive) if inactive else 0.)
        if matches:out[w,[0,1,3]]=[np.mean(matches),np.mean(misses),np.mean(closures)];valid[w,[0,1,3]]=True
        if k[w].any():out[w,2]=unintended;valid[w,2]=True
    return {'terms':out,'valid':valid,'names':['contact_wrench_support','missed_contact','unintended_contact','force_closure']}

def bridge_mujoco_cpu_contacts(solver,contacts):
    """Project CPU readout bridge, matching Newton's installed MJWarp converter.

    This is host transfer for the bounded probe, not a production/GPU bridge.
    Native contact force acts on geom2; Newton stores force on shape0/geom1.
    """
    import mujoco
    m=solver.mj_model;d=solver.mj_data;count=int(d.ncon);capacity=contacts.rigid_contact_max
    if count>capacity:raise ValueError('Contact buffer too small')
    mapping=solver.mjc_geom_to_newton_shape.numpy()[0];s0=np.full(capacity,-1,np.int32);s1=s0.copy();p0=np.zeros((capacity,3),np.float32);p1=p0.copy();normal=p0.copy();force=np.zeros((capacity,6),np.float32)
    for i,c in enumerate(d.contact[:count]):
        s0[i]=mapping[c.geom1];s1[i]=mapping[c.geom2]
        if min(s0[i],s1[i])<0:raise ValueError('Unmapped physical geometry')
        N=np.asarray(c.frame).reshape(3,3);normal[i]=N[0];b0=m.geom_bodyid[c.geom1];b1=m.geom_bodyid[c.geom2];w0=c.pos-.5*c.dist*N[0];w1=c.pos+.5*c.dist*N[0]
        p0[i]=(w0-d.xpos[b0])@d.xmat[b0].reshape(3,3) if b0 else w0;p1[i]=(w1-d.xpos[b1])@d.xmat[b1].reshape(3,3) if b1 else w1
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);force[i,:3]=-N.T@f[:3];force[i,3:]=-N.T@f[3:]
    contacts.rigid_contact_count.assign(np.array([count],np.int32));contacts.rigid_contact_shape0.assign(s0);contacts.rigid_contact_shape1.assign(s1);contacts.rigid_contact_point0.assign(p0);contacts.rigid_contact_point1.assign(p1);contacts.rigid_contact_normal.assign(normal);contacts.force.assign(force)
    return {'native_contact_count':count,'min_native_contact_distance_m':float(min((c.dist for c in d.contact[:count]),default=0)),'max_contact_force_norm_N':float(np.linalg.norm(force[:count,:3],axis=1).max(initial=0.))}
