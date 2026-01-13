## Security Compliance Guidelines

- Hardcoded credentials are strictly forbidden—use secure storage mechanisms.
- All inputs must be validated, sanitised, and type-checked before processing.
- Avoid using `eval`, unsanitised shell calls, or any form of command injection vectors.
- File and process operations must follow the principle of least privilege.
- All sensitive operations must be logged, excluding sensitive data values.
- Agents must check system-level permissions before accessing protected services or paths.
