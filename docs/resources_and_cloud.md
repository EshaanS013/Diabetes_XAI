# Resources needed to continue

This project is a research + mobile screening pipeline. Below is what you need **locally** and in the **cloud** to finish Phase 1–5.

## Local resources (required)

| Resource | Why | Notes |
|---|---|---|
| **Git** | Version control + push/pull | Installing now for GitHub connectivity |
| **GitHub account** | Remote backup + teammate access | Recommended host on Windows |
| **Python 3.10–3.12** (3.13 works with binary wheels) | Training, API, tests | Already have 3.13 + `.venv` |
| **~8–16 GB RAM** | BRFSS training + SMOTE + Optuna | 16 GB comfortable for full-data runs |
| **~5–10 GB disk** | Dataset, artifacts, models, venv | Raw CSV ~12 MB; artifacts grow with experiments |
| **CPU** (modern multi-core) | Tabular ML; no GPU required for Phase 1 | GPU optional, usually unnecessary |
| **Flutter SDK** | Mobile app (Android/iOS) | Not installed yet on this machine |
| **Android Studio / Xcode** | Emulator / device builds | Android Studio on Windows; Xcode needs Mac |
| **Cursor IDE** | Continue agent/dev workflow | Already in use |

### Local software checklist

1. Git + GitHub CLI (`gh`) authenticated  
2. Python venv with `requirements/base.txt`  
3. Dataset via `python scripts/download_dataset.py`  
4. Flutter SDK when starting mobile polish  
5. (Optional) Docker Desktop for API container runs  

---

## Cloud resources (recommended / later)

Aligned with the proposal: **FastAPI + Flutter + Firebase**, with **optional AWS** for compute/deployment.

### Tier A — minimum for September panel (can stay mostly local)

| Cloud | Needed? | Purpose | Cost posture |
|---|---|---|---|
| **GitHub** | **Yes** | Code backup, collaboration, Actions (optional CI) | Free for private student repos |
| **AWS** | Optional | Heavier full-data training if laptop is slow | Student credits (NMIMS / personal) |
| Firebase | Optional for panel | Auth + prediction history | Free Spark tier for demos |
| Domain / CDN | No | Not needed for Phase 1 | — |

### Tier B — deployable mobile screening demo

| Service | Role | Suggested setup |
|---|---|---|
| **AWS (or similar)** | Host FastAPI | ECS/Fargate, App Runner, or a small EC2; **Docker image already in `api/Dockerfile`** |
| **Firebase** | Auth + history sync | Email/Google auth; Firestore/Realtime DB for prediction history metadata |
| **Secrets Manager / Parameter Store** | API keys, Firebase creds | Never commit secrets |
| **S3** (optional) | Model artifact store | Versioned `model.joblib` + metadata |
| **CloudWatch / basic logs** | API latency & errors | Needed once public demo is live |

### Tier C — research compute (only if full-data Optuna is too slow locally)

| Resource | Spec guidance |
|---|---|
| AWS EC2 | `c6i.2xlarge` or similar CPU instance (8 vCPU, 16 GB) for overnight full Phase-1 |
| Spot instances | OK for experiments; save artifacts to S3 before interrupt |
| GPU | **Not required** for these tabular models |

---

## What you do **not** need yet

- GPUs / SageMaker training clusters  
- Kubernetes  
- HIPAA-compliant production stack (do **not** claim compliance without assessment)  
- Public internet domain (localhost + emulator is enough for Phase 1 demos)

---

## Security rules (AWS / Firebase)

1. Enable MFA on cloud root accounts  
2. No root access keys  
3. IAM least privilege for deploy roles  
4. Secrets only in env / secret managers — never in git  
5. Billing alarms / student credit budgets  

See also: `docs/aws_setup.md`, `docs/architecture.md`.

---

## Suggested connectivity setup (this machine)

Because **Cursor Origin repos are not supported on native Windows yet**, use:

1. **GitHub private repo** ← primary remote (full push/pull connectivity)  
2. Continue developing in Cursor against that clone  
3. Later (optional): deploy API to AWS; point Flutter `API_BASE_URL` at the HTTPS endpoint  

---

## Resume order after connectivity

1. Confirm the five Phase-1 algorithms with supervisor  
2. Full-data official Phase-1 train (`configs/phase1_models.yaml` unlocked)  
3. Freeze model + model card  
4. Install Flutter SDK → run `mobile/flutter_app`  
5. (Optional) Firebase auth + AWS API deploy for demo day  
