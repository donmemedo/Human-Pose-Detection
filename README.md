# Human Pose Detection & Movement Analysis System

A comprehensive AI-powered system for real-time human pose detection and movement analysis in videos. This project combines computer vision, deep learning, and a RESTful API to provide detailed human movement analytics.

## 🌟 Features

- **Real-time Pose Detection**: Accurate human pose estimation using MediaPipe
- **Movement Analysis**: Comprehensive analysis of body movements and pose types
- **Video Processing**: Full video analysis with annotated output
- **RESTful API**: FastAPI backend with complete CRUD operations
- **Background Processing**: Asynchronous job handling for large files
- **Multiple Output Formats**: JSON analysis results and annotated videos
- **Job Management**: Track and manage analysis jobs

## 🏗️ System Architecture

```
Input Video → FastAPI Server → Pose Detection AI → Analysis Results
     ↓              ↓               ↓              ↓
   Upload        Job Management   MediaPipe      JSON Report
     ↓              ↓               ↓              ↓
   Storage       Status Tracking   OpenCV      Annotated Video
```

## 📋 Prerequisites

- Python 3.12
- pip (Python package manager)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd human-pose-detection
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

1. **Start the FastAPI server**
```bash
uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the API documentation**
   - Open your browser and go to: http://localhost:8000/docs
   - Interactive Swagger UI with full API documentation

3. **Upload and analyze a video**
```bash
curl -X POST "http://localhost:8000/analyze-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4" \
  -F "generate_output_video=true"
```

## 📁 Project Structure

```
human-pose-detection/
├── pose_detection_ai.py          # AI processing and pose detection
├── fastapi_backend.py            # FastAPI server and endpoints
├── requirements.txt              # Python dependencies
├── uploads/                      # Uploaded video storage
├── outputs/                      # Processed video output
├── results/                      # Analysis JSON results
└── README.md                     # This file
```

## 🔧 API Endpoints

### Video Analysis
- `POST /analyze-video` - Upload video for analysis
- `GET /job-status/{job_id}` - Check analysis job status
- `GET /results/{job_id}` - Download analysis results
- `GET /download/video/{job_id}` - Download annotated video

### Job Management
- `GET /jobs` - List all analysis jobs
- `DELETE /job/{job_id}` - Delete job and associated files
- `GET /health` - System health check

## 📊 Analysis Output

The system provides comprehensive analysis including:

### Pose Detection
- 33 body landmark points
- Skeleton connections
- Bounding boxes
- Visibility scores

### Movement Analysis
- Pose type classification (standing, sitting, arms_raised, etc.)
- Body symmetry scoring
- Activity level assessment
- Frame-by-frame tracking

### Video Analysis Summary
- Total human detections
- Pose type distribution
- Video duration and resolution
- Processing statistics

## 🎮 Usage Examples

### Python Client Example
```python
import requests

# Upload video for analysis
url = "http://localhost:8000/analyze-video"
files = {"file": open("test_video.mp4", "rb")}
response = requests.post(url, files=files, data={"generate_output_video": True})

# Check job status
job_id = response.json()["job_id"]
status_url = f"http://localhost:8000/job-status/{job_id}"
status = requests.get(status_url).json()

# Download results
if status["status"] == "completed":
    results = requests.get(f"http://localhost:8000/results/{job_id}")
    with open("analysis_results.json", "wb") as f:
        f.write(results.content)
```

### Command Line Usage
```bash
# Analyze video and get results
curl -X POST "http://localhost:8000/analyze-video" \
  -F "file=@input_video.mp4" \
  -F "generate_output_video=true"

# Monitor job progress
curl "http://localhost:8000/job-status/YOUR_JOB_ID"

# Download analysis results
curl "http://localhost:8000/results/YOUR_JOB_ID" -o analysis.json
```

## 🔬 AI Models & Technologies

### Core Technologies
- **MediaPipe Pose**: Real-time human pose estimation
- **OpenCV**: Video processing and computer vision
- **TensorFlow**: Deep learning framework
- **FastAPI**: Modern Python web framework

### Pose Detection Features
- 33 key body points detection
- Real-time processing capability
- Robust to various lighting conditions
- Multi-person support (basic)

## ⚙️ Configuration

### Environment Variables
The system can be configured using environment variables:

```bash
export MAX_FILE_SIZE=100000000  # 100MB max file size
export WORKERS=4                # Number of worker processes
export HOST=0.0.0.0            # Server host
export PORT=8000               # Server port
```

### Performance Tuning
- Adjust `model_complexity` in `HumanPoseDetector` for speed/accuracy trade-off
- Modify frame sampling rate in `VideoPoseAnalyzer` for faster processing
- Configure background task workers based on available CPU cores

## 🐛 Troubleshooting

### Common Issues

1. **Video file too large**
   - Solution: Increase `MAX_FILE_SIZE` or compress video

2. **Memory issues during processing**
   - Solution: Reduce video resolution or use frame sampling

3. **Pose detection accuracy low**
   - Solution: Ensure good lighting and clear view of subjects

4. **API timeout**
   - Solution: Use background jobs and check status periodically

### Logs and Debugging
- Check server logs for detailed error information
- Use `/health` endpoint to verify system status
- Monitor job status through API endpoints

## 📈 Performance Metrics

- Processing speed: ~15-30 FPS (depending on hardware)
- Accuracy: >90% for clear frontal poses
- Support: MP4, AVI, MOV video formats
- Maximum resolution: 4K (recommended 1080p for optimal performance)

## 🔮 Future Enhancements

- [ ] Multi-person pose tracking
- [ ] Advanced activity recognition
- [ ] Real-time webcam support
- [ ] Mobile app integration
- [ ] Cloud deployment ready
- [ ] Database integration for results
- [ ] User authentication and management
- [ ] Advanced analytics dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation models
- OpenCV community for computer vision tools
- FastAPI for excellent web framework
- TensorFlow for deep learning infrastructure

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check API documentation at `/docs` endpoint
- Review troubleshooting section in this README

---

**Note**: This system is designed for research and development purposes. Always ensure proper permissions when processing videos containing people.