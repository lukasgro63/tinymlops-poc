from tinysphere.api.routes.dashboard import router as dashboard_router
from tinysphere.api.routes.devices import router as devices_router
from tinysphere.api.routes.models import router as models_router
from tinysphere.api.routes.packages import router as packages_router

__all__ = ["devices_router", "packages_router", "dashboard_router", "models_router"]
