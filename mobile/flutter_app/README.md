# Flutter client — Diabetes XAI

Screening-aid mobile UI (patient + doctor views) calling the FastAPI backend.

**Flutter SDK is not detected on this machine yet.** Install Flutter, then:

```bash
cd mobile/flutter_app
flutter create . --project-name diabetes_xai
flutter pub get
flutter run \
  --dart-define=API_BASE_URL=http://10.0.2.2:8000
```

For iOS simulator / desktop, use `http://127.0.0.1:8000`.

## Screens
- Home with strong non-diagnosis disclaimer
- Questionnaire for 21 BRFSS-style features
- Patient view: risk % + plain-language factors
- Doctor view: model metadata + explanation concordance

## Backend
```bash
# from repo root
.\.venv\Scripts\activate
uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```
