# Universal Capability Compiler laboratory inputs

The JSON files in this directory are conservative policy examples, not
experimental evidence. A complete run uses the following fail-closed sequence:

```text
tidex receiver inspect-safetensors <model-root> <receiver-request.json>
tidex discover capabilities <discovery-request.json>
tidex compile universal-plan <planning-request.json>
tidex materialize {dense|low-rank|sparse|steering} ...
tidex shadow run <authenticated-runner> <shadow-input.json>
tidex select backend <selection-input.json>
tidex measure universality <universality-input.json>
tidex gate promotion <promotion-gate-input.json>
```

The last command never activates a candidate. It can only establish readiness
for TIDE-X's separate promotion authority. Real universality evidence requires
held-out capabilities, multiple receivers and families, multiple seeds, an
unseen receiver, negative controls, preservation measurements, and
`target_optimizer_steps = 0`.
