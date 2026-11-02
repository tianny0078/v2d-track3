"""numpy port of scripts/multiep_recon/deform.py::Deformer (param stage of multiep_recon_v1), for procedural_geometry_v8.

The multiep_recon param-stage deformation is a spatial map of the canonical frame; recon.py exported final_4096.obj as
deform(procedural visual_4096 x sbar). Here the same map is applied to any vertex set (high-res visual, collision parts,
density groups). Frame/constants are rebuilt exactly as Deformer.__init__ does (Vref = init.npz['V'] of that run).
"""
import json
from pathlib import Path
import numpy as np, trimesh

R = Path(__file__).resolve().parents[2]
MR = R/'outputs/track3/multiep_recon_v1'

# adopted run per object (dir holding init.npz + stage_param.pt + report.json); blue_cup final = export_arms P1 cad path
ADOPTED = {
    'plastic_dish_rack': MR/'plastic_dish_rack/variants/B/plastic_dish_rack',
    'white_pot': MR/'white_pot/variants/B/white_pot',
    'beige_cup': MR/'beige_cup/variants/B/beige_cup',
    'white_pot_lid': MR/'white_pot_lid/variants/B/white_pot_lid',
    'wooden_piece_1': MR/'wooden_piece_1/variants/B/wooden_piece_1',
    'wooden_piece_2': MR/'wooden_piece_2/variants/B/wooden_piece_2',
    'wooden_spoon': MR/'wooden_spoon/variants/B/wooden_spoon',
    'blue_cup': R/'outputs/track3/procedural_geometry_v8/blue_cup/source_P1',   # copy of the P1 ablation run (export_arms --runs)
}
NORMALS = {'blue_cup': 'trimesh'}   # export_arms.deform classified inner-wall vertices with trimesh vertex_normals


def area_vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


class DeformNP:
    def __init__(self, kind, pp, Vref, sbar, state, K=5, normals='area'):
        self.kind, self.K, self.sbar, self.normals = kind, K, sbar, normals
        if kind == 'revolve':
            A = np.array(pp['axis'], float); A /= np.linalg.norm(A); C = np.array(pp['centre'], float)*sbar
            h = (Vref - C) @ A; rho = np.linalg.norm((Vref - C) - h[:, None]*A, axis=1)
            self.h0 = pp['h0']*sbar if 'h0' in pp else float(np.percentile(h, 0.2))
            self.h1 = pp['h1']*sbar if 'h1' in pp else float(np.percentile(h, 99.8))
            self.rin = 0.5*(pp['r_bottom_out']*sbar if 'r_bottom_out' in pp else float(np.median(rho)))
            self.use_dt = 't_wall' in pp
            self.A, self.C = A, C
            self.prof = np.asarray(state[0], float).reshape(-1); self.dh = np.asarray(state[1], float).reshape(-1)
            self.dt = float(np.asarray(state[2]).reshape(-1)[0]) if self.use_dt else 0.0
        else:
            c = (Vref.min(0) + Vref.max(0))/2
            ev, E = np.linalg.eigh(np.cov((Vref - c).T)); E = E[:, ::-1].copy()
            ext = np.ptp((Vref - c) @ E, axis=0); E = E[:, np.argsort(-ext)]
            if np.linalg.det(E) < 0: E[:, 2] *= -1
            q = (Vref - c) @ E
            self.lo, self.hi = float(q[:, 0].min()), float(q[:, 0].max()); self.E, self.c = E, c
            self.logs, self.p1, self.p2 = [np.asarray(s, float).reshape(-1) for s in state]

    def _interp(self, p, t):
        K = self.K; x = t*(K-1); i0 = np.clip(np.floor(x), 0, K-2).astype(int); w = x - i0
        return p[i0]*(1-w) + p[i0+1]*w

    def __call__(self, V, F):
        V = np.asarray(V, float)
        if self.kind == 'identity': return V   # pure uniform scale (applied by the caller as x sbar): retrack_v8 sweeper / pitcher
        if self.kind == 'revolve':
            d = V - self.C; h = d @ self.A; rv = d - h[:, None]*self.A; rho = np.linalg.norm(rv, axis=1)
            u = rv/np.maximum(rho, 1e-9)[:, None]
            vn = area_vertex_normals(V, np.asarray(F, np.int64)) if self.normals == 'area' else \
                trimesh.Trimesh(V, np.asarray(F, np.int64), process=False).vertex_normals
            inner = (((vn*u).sum(1) < -0.5) & (rho > self.rin)).astype(float)
            t = np.clip((h - self.h0)/(self.h1 - self.h0), 0, 1)
            ls = self._interp(self.prof, t)
            hn = self.h0 + self.dh[0] + (h - self.h0)*((self.h1 + self.dh[1]) - (self.h0 + self.dh[0]))/(self.h1 - self.h0)
            rn = rho*np.exp(ls) - (inner*self.dt if self.use_dt else 0)
            return self.C + hn[:, None]*self.A + rn[:, None]*u
        q = (V - self.c) @ self.E
        t = np.clip((q[:, 0] - self.lo)/(self.hi - self.lo), 0, 1)
        s1 = np.exp(self.logs[1] + self._interp(self.p1, t)); s2 = np.exp(self.logs[2] + self._interp(self.p2, t))
        qn = np.stack([q[:, 0]*np.exp(self.logs[0]), q[:, 1]*s1, q[:, 2]*s2], 1)
        return self.c + qn @ self.E.T


    def spec(self):
        g = lambda x: np.asarray(x, float).tolist()
        base = dict(kind=self.kind, K=self.K, sbar=self.sbar, normals=self.normals)
        if self.kind == 'revolve':
            return dict(base, A=g(self.A), C=g(self.C), h0=self.h0, h1=self.h1, rin=self.rin, use_dt=self.use_dt, prof=g(self.prof), dh=g(self.dh), dt=self.dt)
        return dict(base, c=g(self.c), E=g(self.E), lo=self.lo, hi=self.hi, logs=g(self.logs), p1=g(self.p1), p2=g(self.p2))

    @classmethod
    def from_spec(cls, s):
        o = cls.__new__(cls); o.kind, o.K, o.sbar, o.normals = s['kind'], s['K'], s['sbar'], s['normals']
        for k, v in s.items():
            if k not in ('kind', 'K', 'sbar', 'normals'): setattr(o, k, np.asarray(v, float) if isinstance(v, list) else v)
        return o


def load_run(obj, state_override=None):
    """-> (DeformNP, sbar, proc_mesh path, proc_params path, pp). Uses the adopted run's report args + init.npz + stage_param.pt."""
    import torch
    d = ADOPTED[obj]; rep = json.loads((d/'report.json').read_text()); a = rep['args']
    pm = Path(a['proc_mesh']); pm = pm if pm.is_absolute() else R/pm
    pj = Path(a['proc_params']); pj = pj if pj.is_absolute() else R/pj
    pp = json.loads(pj.read_text()); init = np.load(d/'init.npz'); sbar = float(init['sbar'])
    st = torch.load(d/'stage_param.pt', weights_only=False, map_location='cpu')
    kind = a.get('deform') or 'auto'
    kind = kind if kind != 'auto' else ('revolve' if 'axis' in pp and 'centre' in pp else 'box')
    if state_override is not None: state = state_override
    elif 'deform_state' in st: state = st['deform_state']
    else: state = [np.log(st['prof']), np.asarray(st['dh']), np.asarray([st['dt']])]   # older stage_param format (revolve only)
    return DeformNP(kind, pp, init['V'], sbar, state, int(a.get('n_prof') or 5), NORMALS.get(obj, 'area')), sbar, pm, pj, pp, st
