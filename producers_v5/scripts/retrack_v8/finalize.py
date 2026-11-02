"""Add v7-compatible fields to the retrack_v8 pose files and write outputs/track3/retrack_v8/READY.json for the learning_v8 agent.
world_T_canonical maps the canonical frame of final_4096.obj (= mesh_scale x mesh_path vertices, verified < 1e-6 m) to the ORB world."""
import json, time
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
R = Path(__file__).resolve().parents[2]; T3 = R/'outputs/track3'; O = T3/'retrack_v8'; MR = T3/'multiep_recon_v1'
pub = json.loads((MR/'eval/public_scans_retrack_v8.json').read_text())
ready = dict(version=2, changelog=['v1 2026-10-31 18:2x UTC: sweeper scored-init mesh + poses; pitcher poses with procedural shape x1.0197',
             'v2: water_pitcher mesh replaced by a scored-only shape re-init (retrack_v8/pitcher_scored_init_B: in-plane warp + thickness field of the '
             'water_pitcher.py bas-relief model, 8x8 grid) x1.0197; ep20 poses re-refined with it (median shift 3.4 deg / 4.7 mm); mini_sweeper unchanged'],
             created_utc=time.strftime('%Y-%m-%d %H:%M', time.gmtime()), producer='retrack_v8 (scripts/retrack_v8, scripts/multiep_recon --init-poses)',
             semantics='world_T_canonical [n,4,4] (+ object_pose_wxyz [n,7] = t xyz + quat wxyz, as learning_v7_geom poses.npz for the slot) '
                       'maps final_4096.obj vertices (metric, scale baked in) to the stereo_sequence_orb_objmask world; frame_index = 0..n-1',
             objects={})
for obj, eps in [('mini_sweeper', [19, 30]), ('water_pitcher', [20])]:
    fin = MR/obj/'final_4096.obj'; Fm = trimesh.load(fin, process=False, force='mesh'); vc = json.loads((MR/obj/'variant_choice.json').read_text())
    poses = {}; trk = {}
    for ep in eps:
        p = O/f'episode_{ep:06d}/{obj}_poses.npz'; z = dict(np.load(p, allow_pickle=True))
        S = trimesh.load(str(z['mesh_path']), process=False, force='mesh'); err = float(np.abs(Fm.vertices - float(z['mesh_scale'])*S.vertices).max())
        assert err < 1e-6, (obj, ep, err)
        W = z['world_T_canonical']; q = Rot.from_matrix(W[:, :3, :3]).as_quat()
        z['object_pose_wxyz'] = np.concatenate([W[:, :3, 3], q[:, [3, 0, 1, 2]]], 1); z['observed'] = np.ones(len(W), bool)
        z['final_mesh_4096'] = str(fin); z['object_names'] = np.array([obj])
        np.savez_compressed(p, **z); poses[f'ep{ep}'] = str(p)
        t = json.loads((O/f'episode_{ep:06d}/{obj}_track.json').read_text())
        trk[f'ep{ep}'] = dict(iou_median_v7_to_new=[t['v7_iou_median'], t['iou_median']], iou_recon_protocol=[t['v7_iou_recon_median'], t['iou_recon_median']],
                              iou_plain=[t['v7_iou_plain_median'], t['iou_plain_median']], depth_mae_mm=[t['v7_depth_mae_mm_median'], t['depth_mae_mm_median']],
                              frac_iou_ge_065=[t['v7_frac_iou_ge_065'], t['frac_iou_ge_065']], n_uncertain=t['n_uncertain'], uncertain_frames=t['uncertain_frames'],
                              report=str(O/f'episode_{ep:06d}/{obj}_track.json'), video=str(O/f'episode_{ep:06d}/{obj}_overlay.mp4'),
                              contact_sheet=str(O/f'episode_{ep:06d}/{obj}_contact_sheet.jpg'))
    ready['objects'][obj] = dict(final_mesh_4096=str(fin), mesh_scale_vs_source=float(z['mesh_scale']), mesh_source=str(z['mesh_path']),
                                 mesh_source_highres=str(Path(str(z['mesh_path'])).parent/'visual.obj'), poses=poses, tracking=trk,
                                 recon_variant_choice=vc, public_cdo_validation_only=pub[obj]['mean'])
sw, pi = ready['objects']['mini_sweeper'], ready['objects']['water_pitcher']
sw['verdict'] = ('ADOPT mesh + poses. Mesh = scored-only re-parametrised procedural sweeper (retrack_v8/sweeper_scored_init: fan 0.435x length, '
                 '1.65x tip width, 1.51x tip thickness, ring 0.99x) x 1.2127; multiep_recon param refit NOT adopted (held-out IoU .793->.793 A / .733 B, '
                 'only 5 edge-on keyframes). Tracking IoU v7 .219/.371 -> .789/.855 (ep19/ep30); public CD-O (validation only) .561 -> .383. '
                 'Collision/URDF must be rebuilt from the new mesh (geometry changed).')
fr = json.loads((O/'pitcher_scored_init_B/fit_report.json').read_text())
pi['shape_fit_heldout'] = dict(iou_known_occluder=[fr['before']['hold_iou'], fr['after']['hold_iou']], depth_mae_mm=[fr['before']['hold_dep'], fr['after']['hold_dep']],
                               n_fit=len(fr['fit_frames']), n_holdout=len(fr['holdout_frames']), protocol='alternating 12-frame time blocks held out; pose-only refit 60 it per frame')
pi['verdict'] = ('ADOPT mesh + poses (v2). Mesh = scored-only shape re-init of the procedural pitcher (pitcher_scored_init_B; held-out IoU '
                 f"{fr['before']['hold_iou']:.3f} -> {fr['after']['hold_iou']:.3f}, depth MAE {fr['before']['hold_dep']:.1f} -> {fr['after']['hold_dep']:.1f} mm) x 1.0197 "
                 '(v7 used procedural x 0.8670). multiep_recon param refit variants still NOT adopted. Tracking IoU v7 .423 -> '
                 f"{trk['ep20']['iou_median_v7_to_new'][1]:.3f} (ep20), depth MAE {trk['ep20']['depth_mae_mm'][0]:.1f} -> {trk['ep20']['depth_mae_mm'][1]:.1f} mm. "
                 f"Public CD-O (validation only): current 1.028 -> {pi['public_cdo_validation_only']['final_4096.obj']:.3f}. Rebuild collision/URDF/mass from the new mesh.")
(O/'READY.json').write_text(json.dumps(ready, indent=1)); print(json.dumps({k: v['verdict'] for k, v in ready['objects'].items()}, indent=1))
