# Data requirements

Each record is a JSON object with a `prompt` and a `completion`.

1. `prompt` must be non-empty and must ask for a sentiment classification.
2. `completion` must be exactly one of: `positive`, `negative`, `neutral`.
3. `completion` must match the sentiment actually expressed in `prompt`.
4. No record may contain personally identifying information.
