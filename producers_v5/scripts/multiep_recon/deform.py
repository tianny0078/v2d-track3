"""Low-dimensional structural deformations of a procedural CAD mesh (param stage of multiep_recon).

revolve : surface-of-revolution objects (cups, pot, lid). Frame = procedural axis/centre. Parameters: log radial scale
          at K heights (piecewise linear in normalised height), bottom/top axial offsets (m), inner-wall radial offset dt
          (m, + = thicker; only vertices whose normal points to the axis). Maps onto the cup generator's r_a/r_b (taper),
          h0/h1, t_wall; marker spheres move with the wall they sit on.
box     : everything else. Frame = principal axes of the CAD mesh (e0 longest). Parameters: log scale along each axis (3)
          + log cross-section scale profiles of e1 and e2 at K stations along e0 (taper / bowl-vs-handle width / thickness).
The same deformation is applied to any vertex set in the canonical frame (dense render mesh, procedural visual_4096).
"""
import math
import numpy as np


class Deformer:
    def __init__(self, cx, kind, pp, Vref, sbar, K=5, Fref=None):
        torch = cx.torch; self.cx = cx; self.kind = kind; self.K = K; dev = cx.dev
        self.sbar = sbar
        if kind == 'revolve':
            A = np.array(pp['axis'], float); A /= np.linalg.norm(A); C = np.array(pp['centre'], float)*sbar
            h = (Vref - C) @ A; rho = np.linalg.norm((Vref - C) - h[:, None]*A, axis=1)
            self.h0 = pp['h0']*sbar if 'h0' in pp else float(np.percentile(h, 0.2))
            self.h1 = pp['h1']*sbar if 'h1' in pp else float(np.percentile(h, 99.8))
            self.rin = 0.5*(pp['r_bottom_out']*sbar if 'r_bottom_out' in pp else float(np.median(rho)))
            self.use_dt = 't_wall' in pp
            self.A, self.C = cx.T(A), cx.T(C)
            self.prof = torch.zeros(K, device=dev, requires_grad=True)
            self.dh = torch.zeros(2, device=dev, requires_grad=True)
            self.dt = torch.zeros(1, device=dev, requires_grad=True)
            self.params = [self.prof, self.dh, self.dt] if self.use_dt else [self.prof, self.dh]
        else:
            c = (Vref.min(0) + Vref.max(0))/2
            ev, E = np.linalg.eigh(np.cov((Vref - c).T)); E = E[:, ::-1].copy()
            ext = np.ptp((Vref - c) @ E, axis=0); E = E[:, np.argsort(-ext)]
            if np.linalg.det(E) < 0: E[:, 2] *= -1
            q = (Vref - c) @ E
            self.lo, self.hi = float(q[:, 0].min()), float(q[:, 0].max())
            self.E, self.c = cx.T(E), cx.T(c)
            self.logs = torch.zeros(3, device=dev, requires_grad=True)
            self.p1 = torch.zeros(K, device=dev, requires_grad=True)
            self.p2 = torch.zeros(K, device=dev, requires_grad=True)
            self.params = [self.logs, self.p1, self.p2]

    def prepare(self, V, F):
        torch = self.cx.torch; Vt = self.cx.T(V)
        if self.kind == 'revolve':
            d = Vt - self.C; h = d @ self.A; rv = d - h[:, None]*self.A[None]; rho = rv.norm(dim=-1); u = rv/rho.clamp_min(1e-9)[:, None]
            Fl = self.cx.T(F, torch.long)
            fn = torch.cross(Vt[Fl[:, 1]]-Vt[Fl[:, 0]], Vt[Fl[:, 2]]-Vt[Fl[:, 0]], dim=-1)
            vn = torch.zeros_like(Vt).index_add_(0, Fl[:, 0], fn).index_add_(0, Fl[:, 1], fn).index_add_(0, Fl[:, 2], fn)
            vn = vn/vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            inner = (((vn*u).sum(-1) < -0.5) & (rho > self.rin)).float()
            tq = ((h - self.h0)/(self.h1 - self.h0)).clamp(0, 1)
            return dict(h=h, rho=rho, u=u, inner=inner, t=tq)
        q = (Vt - self.c) @ self.E
        t = ((q[:, 0] - self.lo)/(self.hi - self.lo)).clamp(0, 1)
        return dict(q=q, t=t)

    def _interp(self, p, t):
        K = self.K; x = t*(K-1); i0 = x.floor().clamp(0, K-2).long(); w = x - i0
        return p[i0]*(1-w) + p[i0+1]*w

    def apply(self, s):
        torch = self.cx.torch
        if self.kind == 'revolve':
            ls = self._interp(self.prof, s['t'])
            h0, h1 = self.h0, self.h1
            hn = h0 + self.dh[0] + (s['h'] - h0)*((h1 + self.dh[1]) - (h0 + self.dh[0]))/(h1 - h0)
            rn = s['rho']*torch.exp(ls) - (s['inner']*self.dt if self.use_dt else 0)
            return self.C + hn[:, None]*self.A[None] + rn[:, None]*s['u']
        q = s['q']
        s1 = torch.exp(self.logs[1] + self._interp(self.p1, s['t'])); s2 = torch.exp(self.logs[2] + self._interp(self.p2, s['t']))
        qn = torch.stack([q[:, 0]*torch.exp(self.logs[0]), q[:, 1]*s1, q[:, 2]*s2], 1)
        return self.c + qn @ self.E.T

    def prior(self):
        sm = lambda p: ((p[1:] - p[:-1])**2).sum()/0.02**2*0.01
        if self.kind == 'revolve':
            return sm(self.prof) + ((self.dt**2).sum()/0.002**2*0.01 if self.use_dt else 0) + (self.dh**2).sum()/0.01**2*0.001
        return sm(self.p1) + sm(self.p2)

    def clamp(self):
        # keep the structural prior sane: +-25 % radial/axial scale, ends move by at most 20 % of the height
        if self.kind == 'revolve':
            self.prof.data.clamp_(-0.25, 0.25); lim = 0.2*(self.h1 - self.h0); self.dh.data.clamp_(-lim, lim)
            if self.use_dt: self.dt.data.clamp_(-0.0012, 0.003)
        else:
            self.logs.data.clamp_(-0.25, 0.25); self.p1.data.clamp_(-0.25, 0.25); self.p2.data.clamp_(-0.25, 0.25)

    def describe(self):
        if self.kind == 'revolve':
            return dict(kind='revolve', radial_scale_profile=np.exp(self.prof.detach().cpu().numpy()).round(4).tolist(),
                        dh_mm=(self.dh.detach().cpu().numpy()*1000).round(2).tolist(),
                        dt_mm=round(float(self.dt.item())*1000, 3) if self.use_dt else None)
        return dict(kind='box', axis_scale=np.exp(self.logs.detach().cpu().numpy()).round(4).tolist(),
                    cross1_scale_profile=np.exp(self.p1.detach().cpu().numpy()).round(4).tolist(),
                    cross2_scale_profile=np.exp(self.p2.detach().cpu().numpy()).round(4).tolist())

    def state(self):
        return [p.detach().cpu().numpy() for p in self.params]

    def load(self, vals):
        for p, v in zip(self.params, vals): p.data.copy_(self.cx.T(v))
