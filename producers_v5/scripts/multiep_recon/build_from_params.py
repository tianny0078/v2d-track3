"""Build a procedural object (scripts/procedural_geometry/<gen>.py build()) from a given params JSON, without fitting to any
existing mesh. Run with .deps/geom-venv/bin/python (manifold3d). Writes visual.obj, visual_4096.obj, params.json, build.json.
Usage: build_from_params.py <generator_module> <params.json> <out_dir>
"""
import importlib, json, sys
from pathlib import Path
import numpy as np, trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'procedural_geometry'))
from build import assemble  # noqa: E402
from common import decimate  # noqa: E402


def main():
    gen, pj, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    mod = importlib.import_module(gen); p = json.loads(pj.read_text())
    vis = assemble(mod.build(p, 'high'))[0]; lo = assemble(mod.build(p, 'low'))[0]
    if len(lo.faces) > 4096:
        d = decimate(lo, 4096); assert d is not None, 'decimation failed'; lo = d
    vis.export(out/'visual.obj'); lo.export(out/'visual_4096.obj'); (out/'params.json').write_text(json.dumps(p, indent=1))
    info = {}
    for k, fn in [('visual', 'visual.obj'), ('visual_4096', 'visual_4096.obj')]:
        r = trimesh.load(out/fn, force='mesh')
        info[k] = dict(faces=len(r.faces), watertight=bool(r.is_watertight), is_volume=bool(r.is_volume), euler=int(r.euler_number),
                       volume_cm3=float(r.volume*1e6), extents_mm=(r.extents*1000).round(1).tolist())
    (out/'build.json').write_text(json.dumps(dict(generator=gen, params=str(pj), **info), indent=1)); print(json.dumps(info))


if __name__ == '__main__':
    main()
