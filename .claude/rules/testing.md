## Testing & Simulation Rules

- All new logic must include unit and integration tests.
- Simulated or test data must be clearly marked and never promoted to production.
- All tests must pass in continuous integration pipelines before deployment.
- Code coverage should exceed defined thresholds (e.g. 85%).
- Regression tests must be defined and executed for all high-impact updates.
- Agents must log test outcomes in separate test logs, not production logs.