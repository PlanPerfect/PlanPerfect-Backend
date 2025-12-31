import os
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from routes.utilities import router as utilities_router
from routes.logger import router as logger_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PlanPerfect Backend", version="1.0.0", swagger_ui_parameters={"defaultModelsExpandDepth": -1})
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(utilities_router)
app.include_router(logger_router)

SERVER_START_TIME = datetime.now()

@app.get("/", tags=["Index"])
def index():
    port = os.getenv("PORT", "8000")
    dev_mode = os.getenv("DEV_MODE", "True").lower() == "true"
    mode_label = "Development" if dev_mode else "Production"


    time_diff = datetime.now() - SERVER_START_TIME
    hours, remainder = divmod(int(time_diff.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        last_reload = f"{hours}h {minutes}m ago"
    elif minutes > 0:
        last_reload = f"{minutes}m {seconds}s ago"
    else:
        last_reload = f"{seconds}s ago"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PlanPerfect Backend</title>
        <link rel="stylesheet" href="/static/serverIndexStyles.css">
    </head>
    <body>
        <div class="background">
            <div class="gradient-orb orb-1"></div>
            <div class="gradient-orb orb-2"></div>
            <div class="gradient-orb orb-3"></div>
        </div>
        <nav class="navbar">
            <div class="nav-content">
                <a href="/" class="logo" style="text-decoration: none;">
                    <span class="logo-text">PlanPerfect Backend</span>
                </a>
                <div class="nav-links">
                    <a href="/" class="nav-link">Home</a>
                    <a href="/docs" class="nav-link">API</a>
                    <a href="/logs" class="nav-link">Logs</a>
                </div>
            </div>
        </nav>
        <main class="container">
            <div class="hero-badge">
                <span class="badge-dot"></span>
                Version 1.0.0
            </div>
            <h1 class="hero-title">
                Welcome to the
                <span class="gradient-text">PlanPerfect Backend</span>
            </h1>
            <div class="cta-group">
                <a href="/docs" class="btn btn-primary">
                    <span>Explore API</span>
                </a>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{port}</div>
                    <div class="stat-label">Server Port</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{last_reload}</div>
                    <div class="stat-label">Last auto-reload</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{mode_label}</div>
                    <div class="stat-label">Server Mode</div>
                </div>
            </div>
        </main>
        <footer class="footer">
            <p>Powered by FastAPI • Developed with ❤️ by the PlanPerfect Team</p>
        </footer>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    load_dotenv()
    port = int(os.getenv("PORT", 8000))
    dev_mode = os.getenv("DEV_MODE", "True").lower() == "true"
    print(f"[{'DEVELOPMENT' if dev_mode else 'PRODUCTION'}] - Server running at http://localhost:{port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=dev_mode,
        log_level="error",
        access_log=False
    )