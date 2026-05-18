# Shared Fine-Tuned Checkpoints

This directory is for fine-tuned checkpoints that should be reused, compared,
or post-processed across VLA backends.

Use one subdirectory per model family, for example:

```text
artifacts/checkpoints/gr00t/<run>/checkpoint-100000
artifacts/checkpoints/starvla/<run>/final_model/pytorch_model.pt
```

Logs, rollout videos, TensorBoard runs, and analysis outputs should remain in
their existing locations unless they are part of the checkpoint payload.
