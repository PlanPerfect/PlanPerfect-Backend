import os
import sys
import signal
import uvicorn
import warnings
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# import routers
from routes.existingHomeOwners.classification import router as style_classification_router
from routes.newHomeOwners.extraction import router as newHomeOwners_extraction_router
from routes.existingHomeOwners.imageGeneration import router as image_router
from routes.newHomeOwners.documentLlm import router as document_llm_router
from routes.chatbot.chatCompletions import router as chatbot_router
from routes.stylematch.detection import router as stylematch_detection_router
from routes.stylematch.reccomendations import router as stylematch_reccomendations_router

from Services import DatabaseManager as DM
from Services import RAGManager as RAG
from Services import LLMManager as LLM
from Services import FileManager as FM
from Services import Bootcheck

warnings.simplefilter("ignore", FutureWarning)

load_dotenv()

app = FastAPI(title="PlanPerfect Backend", version="1.0.0", swagger_ui_parameters={"defaultModelsExpandDepth": -1})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("VITE_FRONTEND_URL"), "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(style_classification_router)
app.include_router(newHomeOwners_extraction_router)
app.include_router(image_router)
app.include_router(document_llm_router)
app.include_router(chatbot_router)
app.include_router(stylematch_detection_router)
app.include_router(stylematch_reccomendations_router)

SERVER_START_TIME = datetime.now()

@app.get("/", include_in_schema=False)
def index():
    port = os.getenv("PORT", "8000")
    dev_mode = os.getenv("DEV_MODE", "True").lower() == "true"
    mode_label = "Development" if dev_mode else "Production"

    server_start_iso = SERVER_START_TIME.isoformat()

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
                    <a href="/sample/logs-page" class="nav-link">Logs</a>
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
                    <div class="stat-value" id="uptime">Calculating...</div>
                    <div class="stat-label">Server Uptime</div>
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

        <script>
            // Server start time passed from backend
            const serverStartTime = new Date("{server_start_iso}");

            function updateUptime() {{
                const now = new Date();
                const diff = now - serverStartTime; // difference in milliseconds

                // Convert milliseconds to days, hours, minutes, seconds
                const days = Math.floor(diff / (1000 * 60 * 60 * 24));
                const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
                const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                const seconds = Math.floor((diff % (1000 * 60)) / 1000);

                // Format the uptime string
                let uptimeString = "";
                if (days > 0) {{
                    uptimeString = `${{days}}d ${{hours}}h ${{minutes}}m`;
                }} else if (hours > 0) {{
                    uptimeString = `${{hours}}h ${{minutes}}m ${{seconds}}s`;
                }} else if (minutes > 0) {{
                    uptimeString = `${{minutes}}m ${{seconds}}s`;
                }} else {{
                    uptimeString = `${{seconds}}s`;
                }}

                // Update the display
                document.getElementById('uptime').textContent = uptimeString;
            }}

            // Update immediately and then every second
            updateUptime();
            setInterval(updateUptime, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    load_dotenv()
    port = int(os.getenv("PORT", 8000))
    dev_mode = os.getenv("DEV_MODE", "True").lower() == "true"

    print(f"SERVER MODE: {'DEVELOPMENT' if dev_mode else 'PRODUCTION'}\n")

    if not Bootcheck.run_checks():
        sys.exit(1)

    if not DM._initialized:
        DM.initialize(
            database_url=os.getenv("FIREBASE_DATABASE_URL"),
            credentials_path=os.getenv("FIREBASE_CREDENTIALS_PATH")
        )

    if not RAG._initialized:
        RAG.initialize(
            document_path=os.getenv("RAG_DOCUMENT_PATH"),
            force_reingest=False
        )

    if not LLM._initialized:
        LLM.initialize()

    if not FM._initialized:
        FM.initialize()

    def signal_handler(signum, frame):
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Server running at \033[94mhttp://localhost:{port}\033[0m\n")

    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="error",
            access_log=False
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)