# Third-party notices

The complete upstream libraries and model assets are fetched separately at the revisions in `config/dependencies.lock.json`.

The following project modules retain NVIDIA SPDX copyright and Apache-2.0 declarations and adapt code or fixtures from NVIDIA `video_to_data`, revision `193382a184308f6800331f3db829f76dc4294cfe`:

- `scripts/track3_masked_contact.py`: derived contact objectives, adding explicit validity and denominator masking.
- `scripts/track3_masked_objective.py`: derived tracking evaluator and explicit objective specification, adding validity and symmetric COM support conversion.
- `scripts/train_track3_masked.py`: derived learner setup/entry integration.
- `scripts/verify_track3_masked_objective.py` and `scripts/verify_track3_masked_com.py`: adapted validation harnesses retaining their original declarations.

These project modifications have not been applied to the pinned upstream checkout. Retained declarations identify the original code; project changes do not claim NVIDIA endorsement.

`licenses/NVIDIA-upstream-license.txt` preserves the upstream repository's supplied license text, including Apache-2.0 for source code and CC-BY-4.0 for documentation/skills. `licenses/FlashSAC-NOTICE.txt` preserves the upstream FlashSAC attribution and MIT terms for the externally imported learner. This repository does not redistribute model weights, datasets or the upstream JAX learner implementation.

This notice does not assign a new license to unrelated project-authored files. See each retained source header and upstream dependency for its applicable terms.
