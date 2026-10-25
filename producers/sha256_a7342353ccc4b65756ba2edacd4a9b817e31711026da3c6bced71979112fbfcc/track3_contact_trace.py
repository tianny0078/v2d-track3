"""Optional post-control-step simulator contact readout; never inferred GT contact.

Samples the final substep only. Absence here does not exclude an intervening
substep collision, and measured force does not establish the cause of drift.
"""
import numpy as np


class ContactTrace:
    def __init__(self):
        self.offsets = [0]
        self.rows = []

    def append(self, readout, count, capacity):
        if count < 0 or count > capacity:
            raise ValueError('Simulator contact buffer count outside capacity')
        row = {}
        for name in ('body0_ids', 'body1_ids', 'contact_pos_w', 'contact_normal_w',
                     'contact_force_w', 'contact_torque_w'):
            value = np.asarray(getattr(readout, name))
            expected = (count,) if name.endswith('_ids') else (count, 3)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError('Invalid simulator contact field: '+name)
            row[name] = value.copy()
        self.rows.append(row)
        self.offsets.append(self.offsets[-1] + count)

    def sample(self, env):
        from flash_chord.runtime.contact import read_contacts_w
        contacts = env.contacts
        if getattr(contacts, 'force', None) is None:
            raise ValueError('Contact force attribute missing; cannot label zeros as measured')
        count = int(contacts.rigid_contact_count.numpy()[0])
        capacity = len(contacts.rigid_contact_shape0)
        if not 0 <= count <= capacity:
            raise ValueError('Simulator contact buffer count outside capacity')
        self.append(read_contacts_w(env.model, env.state_0, contacts), count, capacity)

    def write(self, path, env, expected_frames):
        if len(self.rows) != expected_frames:
            raise ValueError('Incomplete contact trace')
        arrays = {name: np.concatenate([row[name] for row in self.rows], axis=0)
                  for name in self.rows[0]}
        # Store the runtime mapping with the trace; positive body IDs alone
        # cannot identify a hand/link after the simulator has exited.
        labels = getattr(env.model, 'body_label', None)
        if labels is not None:
            labels = np.asarray(labels, dtype=str)
            if labels.ndim != 1 or len(labels) != env.model.body_count:
                raise ValueError('Invalid runtime body label mapping')
            arrays['body_labels'] = labels
        np.savez_compressed(path, **arrays, frame_offsets=np.asarray(self.offsets),
                            object_body_ids_w=env.command.body_ids_w.numpy(),
                            shape_body=env.model.shape_body.numpy())


class ContactStepObserver:
    """Delegate exactly one original step, then read its resulting contacts."""
    def __init__(self, delegate, env, trace):
        self.delegate, self.env, self.trace = delegate, env, trace

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def step(self, action):
        result = self.delegate.step(action)
        self.trace.sample(self.env)
        return result
