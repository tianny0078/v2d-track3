"""Copy an explicit result allowlist and build browser-local visualization data."""
import base64,hashlib,json,shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011'
SITE=ROOT/'sites/track3-results-20261015';DIST=SITE/'dist';ASSETS=DIST/'assets'
ASSETS.mkdir(parents=True,exist_ok=True);(DIST/'downloads').mkdir(exist_ok=True)


def read(name):return json.loads((BASE/name).read_text())


def binary(data):return base64.b64encode(np.ascontiguousarray(data).tobytes()).decode('ascii')


def load_mesh(path):
    header,body=path.read_bytes().split(b'end_header\n',1);lines=header.decode('ascii').splitlines()
    n=int(next(l.split()[-1] for l in lines if l.startswith('element vertex ')))
    m=int(next(l.split()[-1] for l in lines if l.startswith('element face ')))
    assert len(body)==12*n+13*m
    vertices=np.frombuffer(body[:12*n],dtype='<f4').reshape(n,3).copy()
    entries=np.frombuffer(body[12*n:],dtype=np.dtype([('count','u1'),('face','<i4',(3,))]));assert np.all(entries['count']==3)
    faces=entries['face'].astype('<u4');assert np.isfinite(vertices).all() and int(faces.max())<n
    normals=np.zeros_like(vertices);a,b,c=vertices[faces[:,0]],vertices[faces[:,1]],vertices[faces[:,2]]
    face_normals=np.cross(b-a,c-a)
    for i in range(3):np.add.at(normals,faces[:,i],face_normals)
    length=np.linalg.norm(normals,axis=1);normals/=np.maximum(length[:,None],1e-12)
    edge=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edge,axis=0,return_counts=True)
    return vertices,faces,normals,{'boundary_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum())}


def main():
    fusion=read('object_surface_fusion/report.json');stereo=read('stereo_scene_quality/report.json');hands=read('hand_sequence_reviewed/report.json')
    tradeoff=read('hand_depth_tradeoff/report.json');tracks=read('object_surface_tracks/report.json')
    assert fusion['state']=='completed' and tracks['state']=='completed'
    modeldata={};meshes=[]
    for name,label,color in [('blue_cup','蓝杯',[.20,.40,.78]),('beige_cup','米色杯',[.77,.68,.47])]:
        source=BASE/f'object_surface_fusion/{name}/observed_surface.ply';verts,faces,normals,quality=load_mesh(source)
        minimum,maximum=verts.min(axis=0),verts.max(axis=0);r=fusion['objects'][name]
        modeldata[name]={'positions':binary(verts.astype('<f4')),'normals':binary(normals.astype('<f4')),'indices':binary(faces.astype('<u4')),
                         'vertexCount':len(verts),'indexCount':int(faces.size),'center':((minimum+maximum)/2).tolist(),
                         'extent':float((maximum-minimum).max()),'color':color}
        filename=name+'-observed-surface.ply';shutil.copyfile(source,DIST/'downloads'/filename)
        meshes.append({'id':name,'label':label,'vertices':len(verts),'faces':len(faces),'watertight':r['watertight'],
                       'extents_mm':np.round((maximum-minimum)*1000,1).tolist(),'frames_integrated':r['frames_integrated'],
                       'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'file':'downloads/'+filename,**quality})
    cup_dir=BASE/'agent_cup_reconstruction'
    if (cup_dir/'report.json').exists():
        for name,label,color in [('blue_cup','蓝杯 · 空心候选',[.20,.40,.78]),('beige_cup','米色杯 · 空心候选',[.77,.68,.47])]:
            source=cup_dir/name/'wall_03mm_3000um.ply'
            if not source.exists():continue
            verts,faces,normals,quality=load_mesh(source);raw_extent=verts.max(0)-verts.min(0)
            rotation=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float32)
            verts=verts@rotation.T;normals=normals@rotation.T
            minimum,maximum=verts.min(0),verts.max(0);key=name+'_programmatic'
            modeldata[key]={'positions':binary(verts.astype('<f4')),'normals':binary(normals.astype('<f4')),'indices':binary(faces.astype('<u4')),
                            'vertexCount':len(verts),'indexCount':int(faces.size),'center':((minimum+maximum)/2).tolist(),
                            'extent':float((maximum-minimum).max()),'color':color}
            filename=name+'-hollow-candidate.ply';shutil.copyfile(source,DIST/'downloads'/filename)
            meshes.append({'id':key,'label':label,'kind':'programmatic','vertices':len(verts),'faces':len(faces),'watertight':True,
                           'extents_mm':np.round(raw_extent*1000,1).tolist(),'frames_integrated':None,'wall_mm':3,
                           'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'file':'downloads/'+filename,**quality})
        for filename in ['cup_cross_sections.png','previews/initial_beige_cup_000136.jpg','previews/initial_blue_cup_000068.jpg']:
            source=cup_dir/filename
            if source.exists():shutil.copyfile(source,ASSETS/Path(filename).name)
    (ASSETS/'mesh-data.js').write_text('window.V2D_MESHES='+json.dumps(modeldata,separators=(',',':'))+';\n')
    for relative,filename in [
        ('object_meshes_sam3d_local/mesh_preview.jpg','sam3d-cups.jpg'),
        ('object_meshes_sam3d_local/frame0_metric_overlay.jpg','sam3d-initial-overlay.jpg'),
        ('sam3d_metric_registration/blue_cup/frame0_registered_overlay.jpg','sam3d-blue-registered.jpg'),
        ('sam3d_metric_registration/beige_cup/frame0_registered_overlay.jpg','sam3d-beige-registered.jpg'),
        ('agent_cup_pose_refinement/previews/failure_comparison_000113.jpg','cup-table-penetration-failure.jpg'),
        ('agent_cup_pose_supported/previews/comparison_000113.jpg','cup-table-supported.jpg'),
        ('contact_wrench_probe/contact_wrench_uncertainty.png','contact_wrench_uncertainty.png'),
        ('contact_wrench_probe/contact_wrench_uncertainty.pdf','contact_wrench_uncertainty.pdf')]:
        source=BASE/relative
        if source.exists():shutil.copyfile(source,ASSETS/filename)
    rows=[];sg=read('stereo_sequence/report.json');nn=read('foundation_stereo_sequence/report.json')
    for frame in [0,68,136]:
        stem=f'{frame:06d}';prefix='frame-'+stem
        files={'rgb':BASE/f'calibrated_sequence/frames/{stem}.jpg','cups':BASE/f'object_masks_temporal/frame_{stem}_overlay.jpg',
               'hands':BASE/f'hand_masks_sequence/frame_{stem}/overlay.jpg'}
        for kind,path in files.items():shutil.copyfile(path,ASSETS/f'{prefix}-{kind}.jpg')
        for kind,path,key in [('sgbm',BASE/f'stereo_sequence/depth_rgb/{stem}.npz','depth'),
                              ('learned',BASE/f'foundation_stereo_sequence/frame_{stem}/prediction.npz','depth_rgb')]:
            depth=np.load(path,allow_pickle=False)[key];scaled=np.clip((depth-.25)/.95,0,1);scaled[~np.isfinite(scaled)]=0
            image=cv2.applyColorMap((scaled*255).astype(np.uint8),cv2.COLORMAP_TURBO);image[~np.isfinite(depth)]=0
            cv2.imwrite(str(ASSETS/f'{prefix}-{kind}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94])
        rows.append({'frame':frame,'prefix':prefix,'valid':{'sgbm':sg['frames'][frame]['stereo_valid_fraction'],'learned':nn['frames'][frame]['stereo_valid_fraction']},
                     'cupCoverage':{name:{'sgbm':sg['frames'][frame]['objects'][name]['mask_coverage'],
                                          'learned':nn['frames'][frame]['objects'][name]['mask_depth_coverage']} for name in ['blue_cup','beige_cup']}})
    for filename in ['sequence_quality.png','sequence_quality.pdf','three_frame_scene.png','three_frame_scene.pdf']:
        shutil.copyfile(BASE/'stereo_scene_quality'/filename,ASSETS/filename)
    shutil.copyfile(BASE/'object_meshes_triposr/mesh_preview.jpg',ASSETS/'triposr-candidate.jpg')
    stamp=datetime.now(ZoneInfo('America/Los_Angeles'))
    data={'updated':stamp.strftime('%Y-%m-%d %H:%M PDT'),'episode':'episode_000011','frames':137,'handInstances':hands['total_hands'],
          'twoHandFrames':hands['two_hand_frames'],'singleHandFrames':hands['one_hand_frames'],'meshes':meshes,'keyframes':rows,
          'stereoValid':stereo['stereo_valid_fraction_p5_p50_p95'],'objectCoverage':stereo['objects'],
          'tradeoff':list(tradeoff['alphas'].values()),'fusionSeconds':fusion['elapsed_seconds'],
          'fusionCompleted':datetime.fromisoformat(fusion['completed_utc']).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PDT'),
          'scope':'公开样本 episode 11；当前结果不代表 20 段评分数据已完成。',
          'interpretation':'覆盖率、轮廓重合和预测深度差用于内部一致性检查，不等于 GT 准确率。',
          'stages':[{'name':'数据与标定核查','detail':'三路视频，各 137 帧；校准几何一致。'},
                    {'name':'双杯时序分割','detail':'137 帧双对象 mask；保留独立逐帧分割的身份混淆对照。'},
                    {'name':'学习式立体深度','detail':'137 帧；左右一致性检查；与 SGBM 对照。'},
                    {'name':'手部重建与身份修复','detail':'233 个手实例；缺失手保持缺失；原始和修正结果均保留。'},
                    {'name':'深度 / 2D / 连续性折中','detail':'三档试验完成；中档 alpha 0.50 作为候选保留。'},
                    {'name':'双杯表面跟踪与 TSDF 融合','detail':'两杯各融合 137 帧；体积 checkpoint 已保存；输出局部网格。'}],
          'next':['检查局部网格的漂移、碎片和缺口，完善自建物体几何。','将可见表面参考系注册到完整网格，建立可靠物体 6D 轨迹。','联合优化手物接触与碰撞，再进入策略训练和真实 rollout。']}
    (ASSETS/'data.js').write_text('window.V2D_RESULTS='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    (DIST/'downloads/results-summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'notes/track3-fusion-quality.json').write_text(json.dumps({'state':'completed','meshes':meshes,'interpretation':'TSDF surfaces remain open. Programmatic hollow candidates are watertight, but geometry/pose quality has not passed.'},ensure_ascii=False,indent=2)+'\n')
    print('Built curated result data and geometry:',len(meshes),'meshes,',len(rows),'keyframes')


if __name__=='__main__':main()
