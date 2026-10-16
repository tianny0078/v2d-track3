"""Post-pilot geometric mating audit. No physics engine or GT claims."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from agent_cup_reconstruct import OUT,NAMES,THICKNESSES,save,now

def penetration(q,s,t):
    z=q[:,2];rho=np.linalg.norm(q[:,:2],axis=1);r=s['bottom_outer_radius_m']+s['taper_dr_dz']*z
    inside=(z>0)&(z<s['height_m'])&(rho<r)&((z<.003)|(rho>r-t))
    outer=(r-rho)/np.sqrt(1+s['taper_dr_dz']**2)
    inner=(rho-(r-t))/np.sqrt(1+s['taper_dr_dz']**2)
    cavity=np.where(rho<r-t,.003-z,np.maximum(inner,.003-z))
    distance=np.minimum.reduce([z,s['height_m']-z,outer,cavity])
    return np.where(inside,np.maximum(0,distance),0)

def sample_outer(s):
    z=np.linspace(.001,s['height_m']-.001,48);theta=np.arange(128)*2*np.pi/128
    r=s['bottom_outer_radius_m']+s['taper_dr_dz']*z
    return np.stack([r[:,None]*np.cos(theta),r[:,None]*np.sin(theta),np.broadcast_to(z[:,None],(48,128))],axis=-1).reshape(-1,3)

def main():
    r=json.loads((OUT/'report.json').read_text());poses=np.load(OUT/'candidate_poses.npz')
    shapes=[r['objects'][n]['shape'] for n in NAMES];records=[]
    for variant,key in [('initial','world_T_canonical_initial'),('train_refined','world_T_canonical_refined')]:
        P=poses[key]
        for f in [0,20,45,58,60,62,68,105,122,124,125,126,136]:
            T=np.linalg.inv(P[f,0])@P[f,1]
            angle=float(np.rad2deg(np.arccos(np.clip(T[2,2],-1,1))))
            for t in THICKNESSES:
                q=sample_outer(shapes[1])@T[:3,:3].T+T[:3,3]
                pen=penetration(q,shapes[0],t)
                records.append({'variant':variant,'frame':f,'wall_thickness_m':t,'cup_axis_angle_deg':angle,'beige_bottom_in_blue_m':T[:3,3].tolist(),'beige_base_eccentricity_m':float(np.linalg.norm(T[:2,3])),'insertion_along_blue_axis_m':float(shapes[0]['height_m']-T[2,3]),'beige_side_samples_inside_blue_material':int((pen>0).sum()),'sample_count':len(pen),'sampled_max_material_penetration_m':float(pen.max())})
    audit={'created_utc':now(),'method':'Analytic material occupancy of sampled beige outer wall in blue shell, using unchanged candidate poses; not Newton collision validation or exhaustive intersection test.','records':records}
    save(OUT/'nesting_audit.json',audit)
    # Explicit candidate COM transforms, assuming uniform-density shell material.
    coms=[]
    for oi,name in enumerate(NAMES):
        per=[]
        for c in r['objects'][name]['candidates']:
            T=np.eye(4);T[:3,3]=c['uniform_material_COM_canonical_m'];per.append(T)
            c['canonical_T_uniform_COM']=T.tolist()
        coms.append(per)
    C=np.asarray(coms)
    np.savez_compressed(OUT/'candidate_COM_poses.npz',wall_thickness_m=THICKNESSES,canonical_T_uniform_COM=C,world_T_uniform_COM_initial=poses['world_T_canonical_initial'][:,:,None]@C[None],world_T_uniform_COM_refined=poses['world_T_canonical_refined'][:,:,None]@C[None])
    r['nesting_audit']={'path':'nesting_audit.json','collision_validated':False,'interpretation':'Nonzero overlap flags candidate pose/shape inconsistency; thin material penetration magnitude is not a robust separation distance.'}
    r['revision_history']=[{'revision':'v1','evidence_directory':'pilot_v1_failure','finding':'Moving-frame track contamination inflated beige height to 210.8mm. Geometry and validation retained.'},{'revision':'v2','evidence_directory':'pilot_v2_pre_coordinate_fix','finding':'Static TRAIN-only height resolves gross shape error; final audit found table points lacked camera-to-world transform.'},{'revision':'v3','finding':'Table points explicitly transformed through world_T_rgb[0] before plane fit; raw initial object tracking unchanged. Holdout failures retained.'}]
    r['depth_reliability_definition']='Eroded object mask, dilated-hand exclusion, other-object exclusion, finite range and median/MAD filtering; no GT or calibrated confidence bound; stereo_valid is not reprojected in this pilot.'
    r['validation_resolution']={'stride':2,'image_shape':[384,512]}
    r['quality_thresholds']={'median_heldout_silhouette_iou_min':.90,'median_of_frame_depth_p50_max_m':.005,'median_of_frame_depth_p95_max_m':.015,'minimum_frame_depth_render_coverage':.90,'status':'Proposed engineering gates, not benchmark criteria; full per-frame results retained.'}
    save(OUT/'report.json',r)
    im=Image.new('RGB',(900,510),'white');draw=ImageDraw.Draw(im)
    draw.text((20,12),'Observed-fit hollow cup candidates | cross section | wall thickness is assumed',fill='black')
    for oi,s in enumerate(shapes):
        ox=230+oi*440;oy=445;scale=2300
        draw.text((ox-150,40),NAMES[oi]+f"  height={s['height_m']*1000:.1f}mm",fill='black')
        for t,color in zip(THICKNESSES,[(50,110,210),(210,120,40),(90,170,80)]):
            h=s['height_m'];rr=s['bottom_outer_radius_m'];k=s['taper_dr_dz']
            for sign in [-1,1]:
                p=[(sign*rr,0),(sign*(rr+k*h),h),(sign*(rr+k*h-t),h),(sign*(rr+k*.003-t),.003),(0,.003),(0,0),(sign*rr,0)]
                draw.line([(ox+x*scale,oy-z*scale) for x,z in p],fill=color,width=2)
        draw.text((ox-145,465),'Walls 1.5 / 3 / 5mm; bottom assumed 3mm',fill='black')
    im.save(OUT/'cup_cross_sections.png');print('saved nesting audit, COM transforms, cross section')

if __name__=='__main__':main()
