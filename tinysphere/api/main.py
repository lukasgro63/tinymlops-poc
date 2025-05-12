# api/main.py
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tinysphere.api.dependencies.db import engine
# Import database models
from tinysphere.db.models import Base

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("package_extracts", exist_ok=True)

# Create database tables
Base.metadata.create_all(bind=engine)

# Wir verwenden direkt den UTCBaseModel mit field_serializer für die Zeitzonenbehandlung
# Das war der Ansatz, der bei unserer ursprünglichen Implementation funktioniert hat

app = FastAPI(
    title="TinySphere API", 
    description="Central MLOps platform for TinyLCM ecosystem",
    docs_url="/docs",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance-Monitoring Middleware hinzufügen
from tinysphere.api.middleware.performance import performance_middleware

app.middleware("http")(performance_middleware)

# Import routes with plural file names
from tinysphere.api.routes.devices import \
    router as devices_router  # Using plural as you renamed the file

app.include_router(devices_router, prefix="/api/devices", tags=["devices"])

# Add other routers (also using plural naming convention)
from tinysphere.api.routes.packages import router as packages_router

app.include_router(packages_router, prefix="/api/packages", tags=["packages"])

from tinysphere.api.routes.dashboard import router as dashboard_router

app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])

@app.get("/api/status")
def read_status():
    return {
        "status": "OK",
        "version": "0.1.0",
        "name": "TinySphere API"
    }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to TinySphere API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

from tinysphere.api.routes.models import router as models_router

app.include_router(models_router, prefix="/api/models", tags=["models"])

# Add notifications router
from tinysphere.api.routes.notifications import router as notifications_router

app.include_router(notifications_router, prefix="/api/notifications", tags=["notifications"])

# Add drift management router
from tinysphere.api.routes.drift import router as drift_router

app.include_router(drift_router, prefix="/api/drift", tags=["drift"])

# Add prediction images router
from tinysphere.api.routes.prediction_images import router as prediction_images_router

app.include_router(prediction_images_router, prefix="/api/prediction-images", tags=["prediction-images"])
