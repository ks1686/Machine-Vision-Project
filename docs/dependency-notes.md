# Dependency constraints: protobuf / MediaPipe

This document records why `protobuf` is pinned below 5.x and why Dependabot
ignores protobuf >=5 and MediaPipe minor/major bumps
(see `.github/dependabot.yml`). Revisit after a MediaPipe Tasks-API port.

## The constraint chain

1. **The security advisory has no 4.x fix.**
   GHSA-7gcm-g887-7qv7 / CVE-2026-0994 (high, DoS via recursion-depth bypass in
   `google.protobuf.json_format.ParseDict`) is patched only in **protobuf
   >= 5.29.6**. The newest 4.x release is 4.25.9 and remains vulnerable per the
   advisory range `< 5.29.6`.

2. **MediaPipe 0.10.x declares `protobuf>=4.25.3,<5`.**
   Any protobuf 5.x/6.x/7.x bump cannot resolve alongside it — this is what made
   Dependabot's protobuf security-update job fail repeatedly.

3. **Forcing protobuf >=5 next to MediaPipe 0.10.x breaks at runtime.**
   Verified empirically with mediapipe 0.10.21 + protobuf 5.29.6 installed side
   by side: `FaceMesh` init fails with a libprotobuf error while parsing its
   embedded graph config —

   ```
   [libprotobuf ERROR .../text_format.cc:335] Error parsing text-format
   mediapipe.CalculatorGraphConfig: 68:22: Expected identifier, got: \
   RuntimeError: Failed to parse: node {...}
   ```

   The C++ layer of mediapipe 0.10.x is built against the protobuf 4 ABI /
   text-format behavior; a 5.x runtime is not just undeclared but genuinely
   incompatible.

4. **MediaPipe >= 0.10.30 removed the legacy API this pipeline uses.**
   The whole codebase (`face_processing.py` and everything importing it) is
   built on `mediapipe.python.solutions.face_mesh.FaceMesh` (478 landmarks,
   `refine_landmarks=True`). In 0.10.30+ the package ships only the Tasks API
   (`mediapipe.tasks`); `mediapipe.python.solutions` is gone entirely.
   Migrating is a deliberate port project, not a version bump.

## Why the risk is accepted

The vulnerable function (`json_format.ParseDict`) parses **untrusted protobuf
JSON**. This pipeline never calls it: it deserializes no external protobuf data
at all — it only runs face landmark graphs locally over webcam frames. There is
no reachable attack path from this repository's code.

Accordingly, Dependabot alert #9 (protobuf 4.25.9) was dismissed as
`tolerable_risk` with this document cited as evidence.

## Path to actually fixing it

Port `face_processing.py` to the MediaPipe **Tasks API**
(`FaceLandmarker`, 478 landmarks incl. iris), then:

- drop the `protobuf` ignore from `.github/dependabot.yml`,
- raise `requires-python` if the newer wheels allow,
- re-open/dismiss history for the old alert as fixed.
