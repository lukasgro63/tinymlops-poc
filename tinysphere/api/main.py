# api/main.py
import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from tinysphere.api.dependencies.db import engine, SessionLocal
# Import database models
from tinysphere.db.models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("package_extracts", exist_ok=True)

# WICHTIG: Wir wählen einen der beiden Ansätze:
# 1. Entweder Alembic-Migrationen (empfohlen für Produktion, verfolgt Schemaänderungen)
# 2. Oder Base.metadata.create_all() (einfacher für Entwicklung, erstellt alle Tabellen neu)

# Wir überprüfen den Environment-Modus
# - Für Entwicklung: Base.metadata.create_all()
# - Für Produktion: Keine automatische Schemaerstellung (Alembic-Migrationen übernehmen das)
if os.getenv("ENVIRONMENT", "").lower() == "development":
    logger.info("DEVELOPMENT MODE: Creating database schema automatically")
    Base.metadata.create_all(bind=engine)
else:
    # In production, we rely solely on Alembic migrations
    # Database schema should already be created by migrations
    logger.info("PRODUCTION MODE: Using Alembic migrations for schema management")
    
    # OPTIONAL: Verifiziere das Schema, um Diagnose zu erleichtern
    try:
        db = SessionLocal()
        try:
            # Überprüfen Sie die wichtigsten Tabellen und Spalten
            tables_query = text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            tables = [row[0] for row in db.execute(tables_query)]
            
            if "devices" in tables:
                # Prüfe, ob Geolokationsspalten existieren
                columns_query = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'devices' AND column_name IN ('latitude', 'longitude', 'geo_accuracy')")
                geo_columns = [row[0] for row in db.execute(columns_query)]
                
                if len(geo_columns) == 3:
                    logger.info("Database check: All geolocation columns exist in devices table")
                else:
                    missing = set(['latitude', 'longitude', 'geo_accuracy']) - set(geo_columns)
                    logger.warning(f"Database check: Missing geolocation columns in devices table: {missing}")
                
            else:
                logger.warning("Database check: 'devices' table not found in database")
                
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database check error: {e}")

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

# Add drift images router
from tinysphere.api.routes.drift_images import router as drift_images_router

app.include_router(drift_images_router, prefix="/api/drift-images", tags=["drift-images"])

# Add operational logs router
from tinysphere.api.routes.operational_logs import router as operational_logs_router

app.include_router(operational_logs_router, prefix="/api/operational-logs", tags=["operational-logs"])
