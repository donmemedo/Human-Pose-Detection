from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import json
import asyncio
from datetime import datetime
import shutil

from pose_detection_ai import VideoPoseAnalyzer, save_analysis_results

app = FastAPI(
    title="Human Pose Detection API",
    description="API for human pose detection and movement analysis in videos",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for analysis jobs
analysis_jobs = {}

class AnalysisJob(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, error
    input_file: str
    output_files: List[str]
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalysisRequest(BaseModel):
    generate_output_video: bool = True
    sample_frames: bool = True

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str
    results_url: Optional[str] = None
    output_video_url: Optional[str] = None

# Create directories for file storage
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Human Pose Detection API", "status": "running"}

@app.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    generate_output_video: bool = True
):
    """
    Upload a video for pose detection and movement analysis
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_filename = f"{job_id}_{file.filename}"
    input_path = os.path.join("uploads", input_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job entry
    job = AnalysisJob(
        job_id=job_id,
        status="pending",
        input_file=input_filename,
        output_files=[],
        created_at=datetime.now().isoformat()
    )
    analysis_jobs[job_id] = job
    
    # Start background analysis
    background_tasks.add_task(
        process_video_analysis,
        job_id,
        input_path,
        generate_output_video
    )
    
    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        message="Video uploaded successfully. Analysis started.",
        results_url=f"/results/{job_id}",
        output_video_url=f"/download/video/{job_id}" if generate_output_video else None
    )

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of an analysis job
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    return job

@app.get("/results/{job_id}")
async def get_analysis_results(job_id: str):
    """
    Get the analysis results for a completed job
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    
    if job.status != "completed":
        return {"status": job.status, "message": "Analysis still in progress"}
    
    # Return results JSON file
    results_path = os.path.join("results", f"{job_id}_analysis.json")
    if os.path.exists(results_path):
        return FileResponse(
            results_path,
            media_type='application/json',
            filename=f"analysis_results_{job_id}.json"
        )
    
    raise HTTPException(status_code=404, detail="Results file not found")

@app.get("/download/video/{job_id}")
async def download_output_video(job_id: str):
    """
    Download the analyzed output video
    """
    video_path = os.path.join("outputs", f"{job_id}_analyzed.mp4")
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type='video/mp4',
            filename=f"analyzed_video_{job_id}.mp4"
        )
    
    raise HTTPException(status_code=404, detail="Output video not found")

@app.get("/jobs")
async def list_jobs():
    """
    List all analysis jobs
    """
    return {
        "total_jobs": len(analysis_jobs),
        "jobs": list(analysis_jobs.values())
    }

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    
    # Delete associated files
    files_to_delete = [
        os.path.join("uploads", job.input_file),
        os.path.join("outputs", f"{job_id}_analyzed.mp4"),
        os.path.join("results", f"{job_id}_analysis.json")
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove job from storage
    del analysis_jobs[job_id]
    
    return {"message": f"Job {job_id} and associated files deleted"}

async def process_video_analysis(job_id: str, input_path: str, generate_output_video: bool):
    """
    Background task to process video analysis
    """
    try:
        job = analysis_jobs[job_id]
        job.status = "processing"
        
        # Initialize analyzer
        analyzer = VideoPoseAnalyzer()
        
        # Set output paths
        output_video_path = None
        if generate_output_video:
            output_video_path = os.path.join("outputs", f"{job_id}_analyzed.mp4")
        
        output_json_path = os.path.join("results", f"{job_id}_analysis.json")
        
        # Perform analysis
        results = analyzer.analyze_video(input_path, output_video_path)
        
        # Save results
        save_analysis_results(results, output_json_path)
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.now().isoformat()
        job.results = results
        
        if generate_output_video:
            job.output_files.append(f"{job_id}_analyzed.mp4")
        
        job.output_files.append(f"{job_id}_analysis.json")
        
    except Exception as e:
        job = analysis_jobs[job_id]
        job.status = "error"
        job.error = str(e)
        print(f"Error processing job {job_id}: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in analysis_jobs.values() if j.status == "processing"])
    }

# Example data model for pose detection results
class PoseDetectionResult(BaseModel):
    frame_number: int
    timestamp: float
    pose_detected: bool
    landmarks_count: int
    movement_analysis: Dict[str, Any]

class VideoAnalysisSummary(BaseModel):
    total_frames: int
    total_human_detections: int
    pose_types: Dict[str, int]
    video_duration: float
    resolution: str

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)