# Third-party notices

The `easyr1/verl` overlay is derived from EasyR1/verl. Original copyright
headers are retained; its Apache-2.0 license is included as `LICENSE-EasyR1`.
MEDSAGE changes include stage-tag standardization, content-to-token alignment,
answer-reward validation, structured-output validation, and a token-count-normalized
weighted PPO objective with symmetric clipping and optional dual clipping disabled
in the MEDSAGE configuration. These implementation changes do not alter earlier
experiment artifacts. The stage
delimiters are `<LOC>`, `<VIS>`, `<KNO>` and `<CON>`; training targets,
prompts and checkpoints must use compatible delimiters.

LLaMA-Factory, EasyR1, DeepSpeed, PyTorch and vLLM are external dependencies
and retain their respective licenses. No MedEvalKit source is redistributed.

A project-wide license for MEDSAGE-owned code has not been selected in this
release. Third-party licensing does not automatically license that code.
