# Architecture

```
Flutter App (patient + doctor views)
        | HTTPS
        v
FastAPI (validation, inference, optional XAI)
        |
        +-- preprocessing + production model artifact
        +-- calibration / threshold metadata
        +-- deterministic explanation templates
        v
Structured JSON response (+ disclaimer)

Firebase: authentication + prediction history (planned; secrets not in repo)
AWS: optional training/compute; API remains portable via Docker
```

## Design constraints

- Inference logic must not hard-depend on AWS SDKs
- Production model loaded once at process start
- Never describe outputs as diagnosis
