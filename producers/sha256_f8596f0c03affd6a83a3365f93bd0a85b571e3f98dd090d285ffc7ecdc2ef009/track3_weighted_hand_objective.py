"""Soft model-reference confidence for hand rewards; GT/force masks stay separate."""
from dataclasses import dataclass, fields
import numpy as np
import warp as wp
from track3_masked_objective import MaskedObjectiveSpec, MaskedTrackingObjective

@wp.kernel
def weighted_hand_reduction(timestep: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float32),
    frames: int, hands: int, keypoints: int,
    keypoint_sum: wp.array(dtype=wp.float32), joint_value: wp.array(dtype=wp.float32),
    keypoint_out: wp.array(dtype=wp.float32), joint_out: wp.array(dtype=wp.float32)):
    world=wp.tid()
    frame=wp.min(wp.max(timestep[world],0),frames-1)
    kp=float(0.0); joint=float(0.0)
    for hand in range(hands):
        weight=weights[frame*hands+hand]
        kp+=weight*keypoint_sum[world*hands+hand]/float(keypoints)
        joint+=weight*joint_value[world*hands+hand]
    keypoint_out[world]=kp/float(hands)
    joint_out[world]=joint/float(hands)

class WeightedHandObjective(MaskedTrackingObjective):
    def evaluate(self,state,timestep,terminated):
        super().evaluate(state,timestep,terminated)
        wp.launch(weighted_hand_reduction,dim=self.world_count,
            inputs=[timestep,self.hand_reference_weight,self.num_frames,self.num_hands,self.num_keypoints,
                self.hand_keypoint_sum,self.hand_joint_value],
            outputs=[self.buffers.hand_keypoints,self.buffers.hand_joint_pos])
        self.buffers.compose(terminated,self.action_rate_l2,self.action_l2)

@dataclass(frozen=True)
class WeightedHandObjectiveSpec(MaskedObjectiveSpec):
    hand_reference_weight: object = None
    def build(self,*args,**kwargs):
        base=super().build(*args,**kwargs)
        result=WeightedHandObjective(**{f.name:getattr(base,f.name) for f in fields(MaskedTrackingObjective)})
        result.__dict__.update(base.__dict__)
        w=np.asarray(self.hand_reference_weight,dtype=np.float32)
        if w.shape!=(base.num_frames,base.num_hands) or not np.isfinite(w).all() or (w<0).any() or (w>1).any():
            raise ValueError('Expected finite per-frame/per-config-side model weights in [0,1]')
        result.hand_reference_weight=wp.array(w.ravel(),dtype=wp.float32,device=base.ref_wrench_support.device)
        return result
