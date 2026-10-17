"""COM-only support view. Never pass this view to tracking/scene construction."""
import numpy as np
from scipy.spatial.transform import Rotation

class COMSupportReference:
    def __init__(self, reference, com, radius, known, sides):
        self.reference = reference
        self.sides = reference.sides if sides is None else sides
        self.com = np.asarray(com, dtype=np.float32)
        self.radius = np.asarray(radius, dtype=np.float32)
        positions = np.asarray(reference.object_body_pos_w())
        bodies = positions.shape[1]
        self.known = np.asarray(known)
        if self.com.shape != (bodies, 3) or not np.isfinite(self.com).all():
            raise ValueError("finite local COM [body,3] required")
        if self.radius.shape != (bodies,) or not np.isfinite(self.radius).all() or (self.radius <= 0).any():
            raise ValueError("positive finite COM radius [body] required")
        if self.known.shape != (len(positions), len(self.sides), bodies) or not np.isin(self.known, [0,1]).all():
            raise ValueError("binary known [frame,hand,body] required")
    def object_body_pos_w(self):
        q = np.asarray(self.reference.object_body_quat_w())
        r = Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4))
        delta = r.apply(np.broadcast_to(self.com, q.shape[:-1]+(3,)).reshape(-1,3))
        return self.reference.object_body_pos_w() + delta.reshape(q.shape[:-1]+(3,))
    def object_body_quat_w(self):
        return self.reference.object_body_quat_w()
    def object_mesh_radius(self):
        return self.radius
    def _contacts(self, side):
        p = np.asarray(self.reference.contact_pos_w(side)).copy()
        n = np.asarray(self.reference.contact_normal_w(side)).copy()
        ids = np.asarray(self.reference.contact_part_ids(side)).copy()
        h = self.sides.index(side)
        for t in range(len(p)):
            for k, body_id in enumerate(ids[t]):
                if body_id < 1 or body_id > len(self.com) or not self.known[t,h,body_id-1]:
                    p[t,k] = 0; n[t,k] = 0; ids[t,k] = 0
                elif not np.isfinite(p[t,k]).all() or not np.isfinite(n[t,k]).all():
                    raise ValueError("known contact contains nonfinite observation")
        return p,n,ids
    def contact_pos_w(self, side): return self._contacts(side)[0]
    def contact_normal_w(self, side): return self._contacts(side)[1]
    def contact_part_ids(self, side): return self._contacts(side)[2]
