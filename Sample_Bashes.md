Package Installation:
```bash
pip install -r requirements.txt

```

Run Backend:
```bash
uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000

```

Curl Sample:
```bash
curl -X POST "http://localhost:8000/analyze-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4" \
  -F "generate_output_video=true"

```

Curl Sample:
```bash
curl "http://localhost:8000/results/YOUR_JOB_ID" --output results.json

```

Curl Sample:
```bash
curl "http://localhost:8000/job-status/YOUR_JOB_ID"

```
