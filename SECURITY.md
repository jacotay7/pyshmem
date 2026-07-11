# Security policy

## Supported versions

Security fixes are made on the latest released minor version. Older releases
may receive a fix only when the maintainer judges a backport to be low risk.

## Reporting a vulnerability

Do not open a public issue for a vulnerability before coordinated disclosure.
Use GitHub's private vulnerability reporting feature on the repository's
Security tab. If that feature is unavailable, contact the maintainer through
the email address associated with the `Jacob Taylor` commits in this repository.

Include a minimal reproduction, affected versions/platforms, impact, and any
known mitigation. Expect acknowledgement within seven days. A disclosure date
will be coordinated after a fix or mitigation is available.

Shared-memory names are not an authorization boundary. CPU payloads and
metadata are accessible to processes with OS permission to open the segments;
lock and CUDA-handle files are created for the current user. Do not use pyshmem
to exchange untrusted data across security principals. See `docs/format.rst`
for the CUDA reduction trust boundary and metadata validation rules.
