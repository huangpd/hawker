# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue in Hawker, please do **not** open a public GitHub issue.

Instead:

1. Prepare a minimal description of the issue and impact.
2. Include reproduction steps if they are safe to share.
3. Share the report privately with the maintainers through your preferred private channel.

If no dedicated security contact has been published yet, treat the project as not accepting public disclosure before maintainer acknowledgement.

## Scope

Security-sensitive areas in this repository include:

- credential handling and environment loading
- browser session reuse and storage state handling
- generated code execution
- network request replay and request header handling
- dependency integrity and supply-chain hygiene

## Operational Guidance

- Never commit `.env`, storage state, or browser profile data.
- Review dependency updates carefully.
- Treat run artifacts as potentially sensitive if tasks involve authenticated sessions.
