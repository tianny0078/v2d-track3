"""Copy/resize multi-episode recon v1 visuals into web/assets/mer/ and write web/assets/mer/index.json for build_pipeline_page.py.
Per object: turntable (before/after), keyframe contact sheet, per-episode overlays, CD-O diagnostic vs public scan (validation only).
"""
import json, re
from pathlib import Path
import cv2
R = Path(__file__).resolve().parents[2]; M = R/'outputs/track3/multiep_recon_v1'; OUT = R/'web/assets/mer'
OBJ = ['plastic_dish_rack', 'white_pot', 'wooden_spoon', 'blue_cup', 'beige_cup', 'mini_sweeper', 'mini_dust_pan', 'water_pitcher', 'white_pot_lid', 'wooden_piece_1', 'wooden_piece_2']
FLOOR = {'white_pot': .175, 'white_pot_lid': .130, 'beige_cup': .137, 'blue_cup': .134, 'plastic_dish_rack': .271, 'mini_sweeper': .084, 'mini_dust_pan': .112, 'water_pitcher': .214, 'wooden_piece_1': .123, 'wooden_piece_2': .126}


def put(src, dst, w):
    im = cv2.imread(str(src)); h = int(im.shape[0]*w/im.shape[1]) if im.shape[1] > w else im.shape[0]
    if im.shape[1] > w: im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT/dst), im, [cv2.IMWRITE_JPEG_QUALITY, 82]); return dst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cdo = json.loads((M/'eval/public_scans_final_v1.json').read_text())
    status = {}
    for line in (M/'STATUS.md').read_text().splitlines():
        m = re.match(r'- (\w+) \| (.*)', line)
        if m: status[m.group(1)] = m.group(2)
    rows = []
    for o in OBJ:
        d = M/o; r = dict(object=o, status=status.get(o, ''), floor=FLOOR.get(o), images={})
        rep = json.loads((d/'report.json').read_text()) if (d/'report.json').exists() else {}
        r['adopted'] = (d/'final_4096.obj').exists() and o != 'mini_dust_pan'
        if o in cdo: r['cdo_current'] = round(cdo[o]['current'], 4); r['cdo_new'] = round(cdo[o].get('final_4096.obj'), 4) if r['adopted'] else None; r['obb'] = cdo[o]['obb']
        elif o != 'wooden_spoon':  # not refit: current value from the procedural_geometry_v1 validation run (same function/seeds)
            r['cdo_current'] = round(json.loads((R/'outputs/track3/procedural_geometry_v1/eval/public_scans.json').read_text())[o]['procedural_mean'], 4); r['cdo_new'] = None
        if o == 'mini_dust_pan' and o in cdo: r['cdo_variantA'] = round(cdo[o]['final_4096.variantA.obj'], 4)
        if (d/'turntable_before_after.png').exists(): r['images']['turntable'] = put(d/'turntable_before_after.png', f'{o}_turntable.jpg', 2000)
        if (d/'keyframes_contact_sheet.jpg').exists(): r['images']['keyframes'] = put(d/'keyframes_contact_sheet.jpg', f'{o}_keyframes.jpg', 1600)
        r['images']['overlays'] = [put(p, f'{o}_{p.stem}.jpg', 1600) for p in sorted(d.glob('overlay_ep*.jpg'))]
        if (M/f'eval/diag_{o}_final.png').exists(): r['images']['diag'] = put(M/f'eval/diag_{o}_final.png', f'{o}_diag.jpg', 900)
        rows.append(r)
    (OUT/'index.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False)); print(len(rows), sum(1 + len(r['images'].get('overlays', [])) for r in rows))


if __name__ == '__main__': main()
