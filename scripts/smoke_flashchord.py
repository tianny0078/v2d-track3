"""Exercise JAX and Warp on the actual selected GPU; no policy training."""
import importlib.metadata
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
import jax
import jax.numpy as jnp
import warp as wp
import newton
import flash_chord

ROOT = Path(__file__).resolve().parents[1]


@wp.kernel
def double_values(values: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    output[tid] = values[tid] * 2.0


def main():
    started = time.monotonic()
    devices = jax.devices()
    assert any(d.platform == 'gpu' for d in devices), devices
    result = jax.jit(lambda x: x @ x)(jnp.eye(16)).block_until_ready()
    np.testing.assert_allclose(np.asarray(result), np.eye(16))
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        values = wp.array(np.arange(32, dtype=np.float32), dtype=wp.float32)
        output = wp.zeros(32, dtype=wp.float32)
        wp.launch(double_values, dim=32, inputs=[values, output])
        np.testing.assert_allclose(output.numpy(), np.arange(32, dtype=np.float32)*2)
        model = newton.ModelBuilder().finalize(device='cuda:0')
        assert model.device.is_cuda
    report = {'state': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'jax_devices': [str(d) for d in devices], 'warp_device': 'cuda:0',
              'versions': {p: importlib.metadata.version(p) for p in
                           ['flash_chord', 'jax', 'jaxlib', 'warp-lang', 'newton', 'mujoco', 'mujoco-warp']},
              'checks': ['JAX GPU JIT matrix multiplication', 'Warp CUDA kernel numeric result', 'Newton CUDA model construction'],
              'elapsed_seconds': time.monotonic()-started, 'policy_training': False}
    (ROOT/'notes/track3-flashchord-smoke.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
