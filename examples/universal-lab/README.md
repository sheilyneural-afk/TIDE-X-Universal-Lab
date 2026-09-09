# Universal Capability Compiler laboratory inputs

Estos JSON de esta carpeta no son resultados de ejecución ni contienen
valores “inventados”: son plantillas de entrada (políticas)
y ejemplos controlados de configuración para correr un flujo experimental real.
La evidencia real se produce en los directorios `output` de la ejecución y queda
atada criptográficamente a cada materialización.

A complete run uses the following fail-closed sequence:

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

### Ejecutable de integración end-to-end

Use this helper script to run all public laboratory phases in one pass:

```bash
./examples/universal-lab/run-lab-e2e.sh \
  <model-root> \
  <receiver-request.json> \
  <discovery-request.json> \
  <planning-request.json> \
  <dense|low-rank|sparse|steering> \
  <receiver-layout.json> \
  <backend-policy.json|-> \
  <steering-layout.json|-> \
  <shadow-runner-path> \
  <shadow-evaluation-input.json> \
  <backend-selection-input-or-policy.json> \
  <universality-input.json> \
  <promotion-gate-request-or-policy.json> \
  <output-directory>

export TIDEX_BIN=tidex
```

All outputs are saved as:

- `00-lab-summary.json`
- `01-receiver-profile.json`
- `02-discovery-report.json`
- `03-shadow-plan.json`
- `04-shadow-plan-replay.json`
- `05-materialization-candidate.json`
- `06-shadow-evaluation-receipt.json`
- `07-backend-selection-receipt.json`
- `08-universality-receipt.json`
- `09-promotion-receipt.json`

For dense and non-steering backends, unused JSON slots can be passed as `-`.
For backends that require files, provide concrete policy/steering payloads.
When a backend-selection policy is provided instead of a full selection input,
the lab command builds the selection input from the shadow evaluation receipt
produced in the same run. When a promotion policy is provided instead of a full
gate request, the lab command builds the gate request from the same run's
selection, universality and shadow receipts.

### Verificación rápida de resultados

After a run, validate outputs with:

```bash
./examples/universal-lab/check-lab-e2e.sh <output-directory>
./examples/universal-lab/check-lab-e2e.sh <output-directory> --strict
```

`--strict` enforces schema equality checks when `jq` is present.
Without `jq`, both scripts still validate that every required output file exists
and is non-empty.
