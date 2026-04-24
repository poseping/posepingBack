import logging

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.session import get_engine
from app.api import admin, assistant, auth, photo_pose, pose, webcam_pose

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="척추Ping API",
    version="1.0.0",
)

# CORS 설정 (dev or production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(pose.router, prefix="/api/pose", tags=["pose"])
app.include_router(webcam_pose.router, prefix="/api/webcam", tags=["webcam"])
app.include_router(photo_pose.router, prefix="/api/photo", tags=["photo"])
app.include_router(assistant.router, prefix="/api/assistant", tags=["assistant"])


@app.on_event("startup")
async def startup():
    from app.services.mediapipe_detector import MediaPipePoseDetector

    model_asset_path = MediaPipePoseDetector.ensure_model_asset()
    logger.info("pose_landmarker_lite.task ready: %s", model_asset_path)

@app.get("/")
async def root():
    return {"message": "백엔드 설정 통합 테스트"}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

@app.get("/health/db")
async def health_db():
    try:
        with get_engine().connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"database": "connected", "result": value}

@app.get("/health/members")
async def health_members():
    try:
        with get_engine().connect() as connection:
            result = connection.execute(text("SELECT COUNT(*) FROM members"))
            count = result.scalar()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"회원 수": count}
