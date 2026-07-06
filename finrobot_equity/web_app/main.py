import os
import sys
import subprocess
import threading
import uuid
import json
import logging
import hashlib
import secrets
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ============== GitHub OAuth Configuration ==============
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "YOUR_GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "YOUR_GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/api/auth/github/callback")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base path for the actual project (nested structure)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_ROOT = os.path.join(PROJECT_ROOT, "core")
SRC_ROOT = CORE_ROOT  # SRC_ROOT points to core directory, scripts are in core/src
OUTPUT_DIR = os.path.join(CORE_ROOT, "output")
CONFIG_DIR = os.path.join(CORE_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "web_app", "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")  # Unified log directory: finrobot_equity/logs/

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure log directory exists

app = FastAPI(
    title="FinRobot Equity Research",
    version="1.0.0",
    docs_url=None,   # disable default CDN-based docs
    redoc_url=None,  # disable default CDN-based redoc
)

# ============== Rate Limiting ==============
# Heavy pipeline endpoints: 5 runs per user per 10 minutes
# Read/status endpoints: 60 per minute
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "web_app", "static")), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "web_app", "templates"))

# ============== Database Integration ==============
from .database.connection import init_db, SessionLocal
from .database import crud
from .database.models import ReportRequest
from .auth import (
    get_current_user, require_auth, create_user_session, delete_user_session,
    authenticate_user, register_user, get_or_create_github_user, 
    change_user_password, init_default_admin
)
from .middleware import RequestLoggerMiddleware
from .admin_routes import router as admin_router

# Initialize database
init_db()
init_default_admin()

# Add middleware for request logging
app.add_middleware(RequestLoggerMiddleware)

# Include admin routes
app.include_router(admin_router)

# ── Serve Swagger UI from local package assets (no CDN required) ──────────
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="FinRobot Equity Research — API Docs",
        swagger_js_url="/static/vendor/swagger-ui-bundle.js",
        swagger_css_url="/static/vendor/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="FinRobot Equity Research — ReDoc",
        redoc_js_url="/static/vendor/redoc.standalone.js",
    )

# Auth Models
class LoginRequest(BaseModel):
    email: str
    password: str
    remember: bool = False

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

# ============== Log Persistence Helpers ==============

def get_log_file_path(task_id: str) -> str:
    """Return the path to the per-task log file."""
    return os.path.join(LOGS_DIR, f"task_{task_id}.log")

def write_log_to_file(task_id: str, message: str):
    """Append a timestamped message to the on-disk log file."""
    log_path = get_log_file_path(task_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        logger.warning(f"Failed to write log to file: {e}")

def read_log_from_file(task_id: str) -> List[str]:
    """Read all log lines from the on-disk log file."""
    log_path = get_log_file_path(task_id)
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        logger.warning(f"Failed to read log from file: {e}")
        return []

def append_task_log(task_id: str, message: str):
    """Write a log line to in-memory store and disk log file."""
    tasks.append_log(task_id, message)
    write_log_to_file(task_id, message)

# ============== Auth Routes ==============

@app.post("/api/auth/login")
async def login(req: LoginRequest, request: Request, response: Response):
    user = authenticate_user(req.email, req.password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create session
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:500]
    session_id = create_user_session(
        user_id=user.id,
        ip_address=ip_address,
        user_agent=user_agent,
        remember=req.remember
    )
    
    # Set cookie
    max_age = 30 * 24 * 60 * 60 if req.remember else 7 * 24 * 60 * 60
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=max_age,
        samesite="lax"
    )
    
    return {"success": True, "user": {"email": user.email, "name": user.name}}

@app.post("/api/auth/register")
async def register(req: RegisterRequest, request: Request, response: Response):
    user = register_user(req.email, req.password, req.name)
    
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Auto login after register
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:500]
    session_id = create_user_session(
        user_id=user.id,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=7 * 24 * 60 * 60,
        samesite="lax"
    )
    
    return {"success": True, "user": {"email": user.email, "name": user.name}}

@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id:
        delete_user_session(session_id)
    
    response.delete_cookie("session_id")
    return {"success": True}

@app.get("/api/auth/me")
async def get_me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"email": user["email"], "name": user["name"]}

class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str

@app.post("/api/auth/change-password")
async def change_password_route(req: ChangePasswordRequest, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # GitHub users cannot change password
    if user.get("provider") == "github" or user["email"].startswith("github:"):
        raise HTTPException(status_code=400, detail="GitHub users cannot change password")
    
    success = change_user_password(user["id"], req.currentPassword, req.newPassword)
    
    if not success:
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    
    return {"success": True, "message": "Password changed successfully"}

# ============== GitHub OAuth Routes ==============

@app.get("/api/auth/github")
async def github_login():
    """Redirect to GitHub OAuth authorization page"""
    if GITHUB_CLIENT_ID == "YOUR_GITHUB_CLIENT_ID":
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured. Please set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.")
    
    github_auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={GITHUB_REDIRECT_URI}"
        f"&scope=user:email"
    )
    return RedirectResponse(url=github_auth_url)

@app.get("/api/auth/github/callback")
async def github_callback(code: str, response: Response):
    """Handle GitHub OAuth callback"""
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    
    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI
            },
            headers={"Accept": "application/json"}
        )
        token_data = token_response.json()
        
        if "error" in token_data:
            raise HTTPException(status_code=400, detail=token_data.get("error_description", "Failed to get access token"))
        
        access_token = token_data.get("access_token")
        
        # Get user info from GitHub
        user_response = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }
        )
        github_user = user_response.json()
        
        # Get user email (might be private)
        email_response = await client.get(
            "https://api.github.com/user/emails",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }
        )
        emails = email_response.json()
        
        # Find primary email
        primary_email = None
        for email_obj in emails:
            if email_obj.get("primary"):
                primary_email = email_obj.get("email")
                break
        
        if not primary_email:
            primary_email = github_user.get("email") or f"{github_user['login']}@github.local"
        
        # Create or update user in database
        user = get_or_create_github_user(
            email=primary_email,
            name=github_user.get("name") or github_user.get("login"),
            avatar_url=github_user.get("avatar_url"),
            github_id=github_user.get("id")
        )
        
        # Create session
        session_id = create_user_session(user_id=user.id)
        
        # Create redirect response with cookie
        redirect_response = RedirectResponse(url="/", status_code=302)
        redirect_response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=7 * 24 * 60 * 60,
            samesite="lax"
        )
        
        return redirect_response

# ============== Page Routes ==============

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    user = get_current_user(request)
    if not user:
        response = RedirectResponse(url="/login", status_code=303)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return templates.TemplateResponse(request, "index.html", {"user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user:
        response = RedirectResponse(url="/", status_code=303)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return templates.TemplateResponse(request, "login.html")

# ============== Chrome DevTools Route ==============

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools configuration request"""
    return Response(content="", status_code=204)

# ============== Task System ==============

class _FileBackedTaskStore:
    """
    Drop-in replacement for the plain dict ``tasks = {}``.

    Task metadata (status, result, user, type) is persisted to a JSON sidecar
    file alongside each task's log file so the state survives server restarts.
    Logs are kept separately in the existing .log files.

    Thread-safety: a per-task threading.Lock prevents races when multiple
    threads update the same task (e.g. the background worker + a status poll).
    """

    def __init__(self, store_dir: str) -> None:
        self._dir = store_dir
        self._cache: dict = {}
        self._locks: dict = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meta_path(self, task_id: str) -> str:
        return os.path.join(self._dir, f"task_{task_id}.json")

    def _lock_for(self, task_id: str) -> threading.Lock:
        with self._global_lock:
            if task_id not in self._locks:
                self._locks[task_id] = threading.Lock()
            return self._locks[task_id]

    def _load(self, task_id: str) -> dict | None:
        """Read task metadata from disk, return None if not found / corrupt."""
        path = self._meta_path(task_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save(self, task_id: str, data: dict) -> None:
        """Persist task metadata dict to disk (logs excluded — stored separately)."""
        path = self._meta_path(task_id)
        tmp  = path + ".tmp"
        try:
            # Write to a temp file then rename for atomicity
            payload = {k: v for k, v in data.items() if k != "logs"}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"TaskStore: failed to persist task {task_id}: {e}")

    # ------------------------------------------------------------------
    # Public dict-like interface
    # ------------------------------------------------------------------

    def __contains__(self, task_id: str) -> bool:
        if task_id in self._cache:
            return True
        return self._load(task_id) is not None

    def __getitem__(self, task_id: str) -> dict:
        if task_id not in self._cache:
            data = self._load(task_id)
            if data is None:
                raise KeyError(task_id)
            data.setdefault("logs", [])
            self._cache[task_id] = data
        return self._cache[task_id]

    def __setitem__(self, task_id: str, value: dict) -> None:
        with self._lock_for(task_id):
            value.setdefault("logs", [])
            self._cache[task_id] = value
            self._save(task_id, value)

    def get(self, task_id: str, default=None):
        try:
            return self[task_id]
        except KeyError:
            return default

    def update_field(self, task_id: str, key: str, value) -> None:
        """Update a single field and persist without replacing the whole dict."""
        with self._lock_for(task_id):
            task = self[task_id]
            task[key] = value
            self._save(task_id, task)

    def append_log(self, task_id: str, message: str) -> None:
        """Append a log line to the in-memory log list (already file-backed via write_log_to_file)."""
        with self._lock_for(task_id):
            if task_id in self._cache:
                self._cache[task_id].setdefault("logs", []).append(message)


tasks = _FileBackedTaskStore(LOGS_DIR)

class AnalysisRequest(BaseModel):
    ticker: str
    company_name: str
    peers: List[str] = []
    years_limit: int = 5
    revenue_growth_2025: float = 0.05
    revenue_growth_2026: float = 0.06
    revenue_growth_2027: float = 0.04
    margin_improvement: float = 0.01
    sga_margin_improvement: float = -0.005
    generate_text: bool = True
    generate_pdf: bool = True
    fmp_api_key: Optional[str] = None       # overrides config.ini if provided
    openai_api_key: Optional[str] = None    # overrides config.ini if provided
    openai_base_url: Optional[str] = None   # optional proxy URL (e.g. SiliconFlow)
    openai_model: Optional[str] = None      # override model (e.g. gpt-4o)
    enable_sensitivity_analysis: bool = True
    enable_catalyst_analysis: bool = True
    enable_enhanced_news: bool = True
    enable_enhanced_charts: bool = True
    enable_valuation_analysis: bool = True
    news_days_back: int = 5
    news_limit: int = 25
    period: str = "annual"
    as_of_date: Optional[str] = None        # anchor date for walk-forward PIT filtering


class WalkforwardRequest(BaseModel):
    tickers: List[str] = ["AAPL"]
    universe: str = "small"                 # "default" | "small" | "custom" (custom uses tickers list)
    start: str
    end: str
    skip_analysis: bool = False
    years_limit: int = 5
    news_days_back: int = 5
    news_limit: int = 25
    price_lookback_days: int = 365
    fmp_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: Optional[str] = None


class ScoreRequest(BaseModel):
    walkforward_root: str
    neutral_band: float = 0.02
    min_pattern_count: int = 2
    skip_price_download: bool = False
    horizons: List[int] = [30, 60, 90]      # IC decay horizon days

def run_process(command, task_id, cwd=None, extra_env: dict = None):
    """Run a shell command and capture output to the task logs.

    extra_env: optional dict of environment variable overrides (e.g. per-request API keys).
    """
    logger.info(f"Task {task_id}: Running command: {' '.join(command)}")
    append_task_log(task_id, f"Executing: {' '.join(command)}")

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if extra_env:
            env.update({k: v for k, v in extra_env.items() if v})

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cwd or SRC_ROOT
        )
        
        for line in process.stdout:
            append_task_log(task_id, line.strip())  
            
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"Command failed with return code {process.returncode}")
            
        return True
    except Exception as e:
        append_task_log(task_id, f"Error: {str(e)}")  
        tasks.update_field(task_id, "status", "failed")
        return False

def execute_analysis_pipeline(task_id: str, req: AnalysisRequest):
    tasks.update_field(task_id, "status", "running")
    append_task_log(task_id, "Starting analysis pipeline...")

    # Update report status in database
    try:
        db = SessionLocal()
        crud.update_report_request(db, task_id, "running")
        db.close()
    except Exception as e:
        logger.warning(f"Failed to update report status: {e}")

    python_exe = sys.executable
    src_dir = os.path.join(SRC_ROOT, "src")
    config_file = os.path.join(CONFIG_DIR, "config.ini")

    # Per-request API key overrides — picked up by common_utils.get_api_key()
    extra_env: Dict[str, str] = {}
    if req.fmp_api_key:
        extra_env["FMP_API_KEY"] = req.fmp_api_key
    if req.openai_api_key:
        extra_env["OPENAI_API_KEY"] = req.openai_api_key
    if req.openai_base_url:
        extra_env["OPENAI_BASE_URL"] = req.openai_base_url
    if req.openai_model:
        extra_env["OPENAI_MODEL"] = req.openai_model

    # Create output directories
    analysis_output_dir = os.path.join(OUTPUT_DIR, req.ticker, "analysis")
    report_output_dir = os.path.join(OUTPUT_DIR, req.ticker, "report")
    os.makedirs(analysis_output_dir, exist_ok=True)
    os.makedirs(report_output_dir, exist_ok=True)

    # Step 1: Generate Financial Analysis
    cmd_analysis = [
        python_exe,
        os.path.join(src_dir, "generate_financial_analysis.py"),
        "--company-ticker", req.ticker,
        "--company-name", req.company_name,
        "--years-limit", str(req.years_limit),
        "--revenue-growth-2025", str(req.revenue_growth_2025),
        "--revenue-growth-2026", str(req.revenue_growth_2026),
        "--revenue-growth-2027", str(req.revenue_growth_2027),
        "--margin-improvement", str(req.margin_improvement),
        "--sga-margin-improvement", str(req.sga_margin_improvement),
        "--news-days-back", str(req.news_days_back),
        "--news-limit", str(req.news_limit),
        "--period", req.period,
        "--output-dir", analysis_output_dir,
    ]

    if req.as_of_date:
        cmd_analysis.extend(["--as-of-date", req.as_of_date])

    if req.peers:
        cmd_analysis.append("--peer-tickers")
        cmd_analysis.extend(req.peers)

    if req.generate_text:
        cmd_analysis.append("--generate-text-sections")
    
    # Enhanced analysis feature flags
    if req.enable_sensitivity_analysis:
        cmd_analysis.append("--enable-sensitivity-analysis")
    if req.enable_catalyst_analysis:
        cmd_analysis.append("--enable-catalyst-analysis")
    if req.enable_enhanced_news:
        cmd_analysis.append("--enable-enhanced-news")
        
    cmd_analysis.extend(["--config-file", config_file])

    if not run_process(cmd_analysis, task_id, cwd=SRC_ROOT, extra_env=extra_env):
        try:
            db = SessionLocal()
            crud.update_report_request(db, task_id, "failed", "Financial analysis failed")
            db.close()
        except Exception as e:
            logger.warning(f"Failed to update report status: {e}")
        return

    # Step 2: Create Equity Report
    base_output_dir = analysis_output_dir

    cmd_report = [
        python_exe,
        os.path.join(src_dir, "create_equity_report.py"),
        "--company-ticker", req.ticker,
        "--company-name", req.company_name,
        "--analysis-csv", os.path.join(base_output_dir, "financial_metrics_and_forecasts.csv"),
        "--ratios-csv", os.path.join(base_output_dir, "ratios_raw_data.csv"),
        "--tagline-file", os.path.join(base_output_dir, "tagline.txt"),
        "--company-overview-file", os.path.join(base_output_dir, "company_overview.txt"),
        "--investment-overview-file", os.path.join(base_output_dir, "investment_overview.txt"),
        "--valuation-overview-file", os.path.join(base_output_dir, "valuation_overview.txt"),
        "--risks-file", os.path.join(base_output_dir, "risks.txt"),
        "--competitor-analysis-file", os.path.join(base_output_dir, "competitor_analysis.txt"),
        "--major-takeaways-file", os.path.join(base_output_dir, "major_takeaways.txt"),
        "--output-dir", report_output_dir,
        "--config-file", config_file,
        "--enable-text-regeneration"
    ]
    
    # Enhanced analysis feature flags
    if req.enable_enhanced_charts:
        cmd_report.append("--enable-enhanced-charts")
    if req.enable_valuation_analysis:
        cmd_report.append("--enable-valuation-analysis")
    
    # Include enhanced analysis file paths in command
    if req.enable_sensitivity_analysis:
        sensitivity_file = os.path.join(base_output_dir, "sensitivity_analysis.json")
        if os.path.exists(sensitivity_file):
            cmd_report.extend(["--sensitivity-analysis-file", sensitivity_file])
    
    if req.enable_catalyst_analysis:
        catalyst_file = os.path.join(base_output_dir, "catalyst_analysis.json")
        if os.path.exists(catalyst_file):
            cmd_report.extend(["--catalyst-analysis-file", catalyst_file])
    
    if req.enable_enhanced_news:
        enhanced_news_file = os.path.join(base_output_dir, "enhanced_news.json")
        if os.path.exists(enhanced_news_file):
            cmd_report.extend(["--enhanced-news-file", enhanced_news_file])
    
    if req.peers:
        cmd_report.extend([
            "--peer-ebitda-csv", os.path.join(base_output_dir, "peer_ebitda_comparison.csv"),
            "--peer-ev-ebitda-csv", os.path.join(base_output_dir, "peer_ev_ebitda_comparison.csv")
        ])

    if not run_process(cmd_report, task_id, cwd=SRC_ROOT, extra_env=extra_env):
        try:
            db = SessionLocal()
            crud.update_report_request(db, task_id, "failed", "Report creation failed")
            db.close()
        except Exception as e:
            logger.warning(f"Failed to update report status: {e}")
        return

    # Step 3: Generate PDF Report
    if req.generate_pdf:
        append_task_log(task_id, "Generating PDF report...")
        cmd_pdf = [
            python_exe,
            os.path.join(src_dir, "generate_pdf_report.py"),
            "--company-ticker", req.ticker,
            "--company-name", req.company_name,
            "--analysis-dir", base_output_dir,
            "--output-dir", report_output_dir,
            "--config-file", config_file,
        ]

        if not run_process(cmd_pdf, task_id, cwd=SRC_ROOT, extra_env=extra_env):
            append_task_log(task_id, "Warning: PDF generation failed, but HTML reports are available.")  

    tasks.update_field(task_id, "status", "completed")
    append_task_log(task_id, "Pipeline completed successfully!")  
    
    # Update report status in database
    try:
        db = SessionLocal()
        crud.update_report_request(db, task_id, "completed")
        db.close()
    except Exception as e:
        logger.warning(f"Failed to update report status: {e}")
    
    # Get report files
    report_files = []
    if os.path.exists(report_output_dir):
        report_files = [f for f in os.listdir(report_output_dir) if f.endswith((".html", ".pdf"))]
    
    # Separate HTML and PDF files — only Professional reports
    html_files = [f for f in report_files if f.endswith('.html')]
    pdf_files = [f for f in report_files if f.endswith('.pdf')]

    # Only Professional HTML; fallback to others only if no Professional exists
    prof_htmls = [f for f in html_files if 'Professional' in f]
    sorted_htmls = prof_htmls if prof_htmls else html_files

    # Only Professional/Equity Report PDFs (exclude chart PDFs like *_ebitda_margin.pdf)
    prof_pdfs = [f for f in pdf_files if 'Professional_Equity_Report' in f]
    if not prof_pdfs:
        prof_pdfs = [f for f in pdf_files if 'Equity_Report' in f]
    sorted_pdfs = prof_pdfs
    
    tasks.update_field(task_id, "result", {
        "report_dir": report_output_dir,
        "ticker": req.ticker,
        "html": sorted_htmls,
        "pdf": sorted_pdfs
    })

def execute_walkforward_pipeline(task_id: str, req: WalkforwardRequest):
    tasks.update_field(task_id, "status", "running")
    append_task_log(task_id, "Starting walk-forward evaluation...")

    python_exe = sys.executable
    src_dir = os.path.join(SRC_ROOT, "src")
    config_file = os.path.join(CONFIG_DIR, "config.ini")
    output_dir = os.path.join(OUTPUT_DIR, "walkforward", task_id)
    os.makedirs(output_dir, exist_ok=True)

    # Per-request API key overrides
    extra_env: Dict[str, str] = {}
    if req.fmp_api_key:
        extra_env["FMP_API_KEY"] = req.fmp_api_key
    if req.openai_api_key:
        extra_env["OPENAI_API_KEY"] = req.openai_api_key
    if req.openai_base_url:
        extra_env["OPENAI_BASE_URL"] = req.openai_base_url
    if req.openai_model:
        extra_env["OPENAI_MODEL"] = req.openai_model

    cmd = [
        python_exe,
        os.path.join(src_dir, "run_walkforward_eval.py"),
        "--start", req.start,
        "--end", req.end,
        "--years-limit", str(req.years_limit),
        "--news-days-back", str(req.news_days_back),
        "--news-limit", str(req.news_limit),
        "--output-root", output_dir,
        "--price-lookback-days", str(req.price_lookback_days),
        "--config-file", config_file,
    ]

    # Ticker vs universe selection: explicit tickers take priority
    if req.tickers and req.tickers != ["AAPL"]:
        cmd.extend(["--tickers", *req.tickers])
    else:
        cmd.extend(["--universe", req.universe])

    if req.skip_analysis:
        cmd.append("--skip-analysis")

    success = run_process(cmd, task_id, cwd=src_dir, extra_env=extra_env)
    if not success:
        tasks.update_field(task_id, "status", "failed")
        return

    tasks.update_field(task_id, "status", "completed")
    append_task_log(task_id, "Walk-forward evaluation completed!")

    summary_file     = os.path.join(output_dir, "summary.json")
    predictions_file = os.path.join(output_dir, "predictions.csv")
    snapshots_file   = os.path.join(output_dir, "snapshots.csv")
    portfolio_file   = os.path.join(output_dir, "portfolio_summary.json")

    result: Dict = {"output_dir": output_dir}
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            result["summary"] = json.load(f)
    if os.path.exists(predictions_file):
        result["predictions_file"] = predictions_file
    if os.path.exists(snapshots_file):
        result["snapshots_file"] = snapshots_file
    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r", encoding="utf-8") as f:
            result["portfolio_summary"] = json.load(f)
    tasks.update_field(task_id, "result", result)


def execute_score_pipeline(task_id: str, req: ScoreRequest):
    tasks.update_field(task_id, "status", "running")
    append_task_log(task_id, "Starting walk-forward scoring...")

    python_exe = sys.executable
    src_dir = os.path.join(SRC_ROOT, "src")
    score_output = os.path.join(req.walkforward_root, "scorecard")

    cmd = [
        python_exe,
        os.path.join(src_dir, "score_walkforward_eval.py"),
        "--walkforward-root", req.walkforward_root,
        "--output-dir", score_output,
        "--neutral-band", str(req.neutral_band),
        "--min-pattern-count", str(req.min_pattern_count),
        "--horizons", *[str(h) for h in req.horizons],
    ]
    if req.skip_price_download:
        cmd.append("--skip-price-download")

    success = run_process(cmd, task_id, cwd=src_dir)
    if not success:
        tasks.update_field(task_id, "status", "failed")
        return

    tasks.update_field(task_id, "status", "completed")
    append_task_log(task_id, "Scoring completed!")

    result: Dict = {"scorecard_dir": score_output}
    report_file      = os.path.join(score_output, "walkforward_report.md")
    scorecard_file   = os.path.join(score_output, "scorecard.json")
    wrong_cases_file = os.path.join(score_output, "wrong_cases.json")
    right_cases_file = os.path.join(score_output, "right_cases.json")

    if os.path.exists(report_file):
        result["report_file"] = report_file
        result["report_text"] = open(report_file, encoding="utf-8").read()
    if os.path.exists(scorecard_file):
        with open(scorecard_file, "r", encoding="utf-8") as f:
            result["scorecard"] = json.load(f)
    if os.path.exists(wrong_cases_file):
        result["wrong_cases_file"] = wrong_cases_file
    if os.path.exists(right_cases_file):
        result["right_cases_file"] = right_cases_file
    tasks.update_field(task_id, "result", result)


@app.post("/api/walkforward/run")
@limiter.limit("5/10minutes")
async def run_walkforward(req: WalkforwardRequest, request: Request, background_tasks: BackgroundTasks):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "logs": [], "result": None, "user": user["email"], "type": "walkforward"}
    write_log_to_file(task_id, f"Walk-forward task created by: {user['email']}")
    background_tasks.add_task(execute_walkforward_pipeline, task_id, req)
    return {"task_id": task_id}


@app.post("/api/walkforward/score")
@limiter.limit("5/10minutes")
async def score_walkforward(req: ScoreRequest, request: Request, background_tasks: BackgroundTasks):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "logs": [], "result": None, "user": user["email"], "type": "score"}
    write_log_to_file(task_id, f"Score task created by: {user['email']}")
    background_tasks.add_task(execute_score_pipeline, task_id, req)
    return {"task_id": task_id}


@app.get("/api/walkforward/list")
async def list_walkforward_runs(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    wf_root = os.path.join(OUTPUT_DIR, "walkforward")
    runs = []
    if os.path.exists(wf_root):
        for entry in sorted(os.listdir(wf_root), reverse=True):
            run_dir = os.path.join(wf_root, entry)
            summary_file = os.path.join(run_dir, "summary.json")
            if os.path.isdir(run_dir) and os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                scorecard_file = os.path.join(run_dir, "scorecard", "scorecard.json")
                scorecard = None
                if os.path.exists(scorecard_file):
                    with open(scorecard_file, "r", encoding="utf-8") as f:
                        scorecard = json.load(f)
                runs.append({
                    "run_id": entry,
                    "summary": summary,
                    "output_dir": run_dir,
                    "scorecard": scorecard,
                })
    return {"runs": runs}


@app.get("/api/walkforward/{run_id}/report")
async def get_walkforward_report(run_id: str, request: Request):
    """Return the markdown scorecard report for a completed walk-forward run."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    report_file = os.path.join(OUTPUT_DIR, "walkforward", run_id, "scorecard", "walkforward_report.md")
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report not found. Run scoring first.")
    return {"run_id": run_id, "report": open(report_file, encoding="utf-8").read()}


@app.get("/api/walkforward/{run_id}/scorecard")
async def get_walkforward_scorecard(run_id: str, request: Request):
    """Return the full JSON scorecard (IC, Sharpe, hit-rate, factor attribution) for a run."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    sc_file = os.path.join(OUTPUT_DIR, "walkforward", run_id, "scorecard", "scorecard.json")
    if not os.path.exists(sc_file):
        raise HTTPException(status_code=404, detail="Scorecard not found. Run scoring first.")
    with open(sc_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/walkforward/{run_id}/cases")
async def get_walkforward_cases(run_id: str, request: Request, outcome: str = "wrong"):
    """Return the right/wrong case-pack for post-mortem analysis."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if outcome not in ("right", "wrong"):
        raise HTTPException(status_code=400, detail="outcome must be 'right' or 'wrong'")
    case_file = os.path.join(OUTPUT_DIR, "walkforward", run_id, "scorecard", f"{outcome}_cases.json")
    if not os.path.exists(case_file):
        raise HTTPException(status_code=404, detail=f"{outcome}_cases.json not found.")
    with open(case_file, "r", encoding="utf-8") as f:
        return {"run_id": run_id, "outcome": outcome, "cases": json.load(f)}


@app.get("/walkforward", response_class=HTMLResponse)
async def walkforward_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "walkforward.html", {"user": user})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "admin.html", {"user": user})


@app.post("/api/run")
@limiter.limit("10/10minutes")
async def run_analysis(req: AnalysisRequest, request: Request, background_tasks: BackgroundTasks):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "logs": [], "result": None, "user": user["email"]}
    
    # Initialise the on-disk log file for this task
    write_log_to_file(task_id, f"Task created by user: {user['email']}")
    write_log_to_file(task_id, f"Ticker: {req.ticker}, Company: {req.company_name}")
    
    # Record report request in database
    try:
        db = SessionLocal()
        crud.create_report_request(
            db=db,
            user_id=user["id"],
            task_id=task_id,
            ticker=req.ticker,
            company_name=req.company_name,
            peers=",".join(req.peers) if req.peers else None,
            generate_text=req.generate_text,
            generate_pdf=req.generate_pdf,
            enable_sensitivity=req.enable_sensitivity_analysis,
            enable_catalyst=req.enable_catalyst_analysis,
            enable_enhanced_news=req.enable_enhanced_news
        )
        db.close()
    except Exception as e:
        logger.warning(f"Failed to record report request: {e}")
    
    background_tasks.add_task(execute_analysis_pipeline, task_id, req)
    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    task = tasks[task_id]
    # Always attach the persisted log lines so the UI gets a full log
    # even after a server restart (in-memory list may be empty then).
    file_logs = read_log_from_file(task_id)
    merged_logs = file_logs if file_logs else task.get("logs", [])
    return {**task, "logs": merged_logs}

# ============== Log File API ==============

@app.get("/api/logs/{task_id}")
async def get_task_logs(task_id: str, request: Request):
    """Return all persisted log lines for a task."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    log_path = get_log_file_path(task_id)
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")
    
    logs = read_log_from_file(task_id)
    return {
        "task_id": task_id,
        "log_file": log_path,
        "logs": logs,
        "line_count": len(logs)
    }

@app.get("/api/logs/{task_id}/download")
async def download_task_logs(task_id: str, request: Request):
    """Download the raw log file for a task."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    log_path = get_log_file_path(task_id)
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")
    
    return FileResponse(
        path=log_path,
        filename=f"task_{task_id}.log",
        media_type="text/plain"
    )

@app.get("/api/logs")
async def list_all_logs(request: Request):
    """List all task log files (admin only)."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Admin-only endpoint
    # Admin emails can be configured via FINROBOT_ADMIN_EMAILS env var (comma-separated)
    admin_emails = os.getenv("FINROBOT_ADMIN_EMAILS", "admin@finrobot.com").split(",")
    admin_emails = [e.strip() for e in admin_emails]
    if user.get("email") not in admin_emails:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    log_files = []
    if os.path.exists(LOGS_DIR):
        for filename in os.listdir(LOGS_DIR):
            if filename.endswith(".log"):
                file_path = os.path.join(LOGS_DIR, filename)
                stat = os.stat(file_path)
                log_files.append({
                    "filename": filename,
                    "task_id": filename.replace("task_", "").replace(".log", ""),
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    # Sort by modification time, newest first
    log_files.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return {
        "logs_dir": LOGS_DIR,
        "total_files": len(log_files),
        "files": log_files
    }

@app.get("/api/history")
async def get_history(request: Request):
    """Return the current user's report history for UI restoration."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        db = SessionLocal()
        reports = crud.get_user_reports(db, user["id"], limit=50)
        result = []
        seen_tickers = set()
        for r in reports:
            # One entry per ticker (latest wins — output files are overwritten)
            if r.ticker in seen_tickers:
                continue
            seen_tickers.add(r.ticker)
            # Verify the report file still exists on disk
            report_dir = os.path.join(OUTPUT_DIR, r.ticker, "report")
            html_files = []
            pdf_files = []
            if os.path.exists(report_dir):
                all_files = os.listdir(report_dir)
                # Include only Professional HTML reports
                prof_html = [f for f in all_files if f.endswith('.html') and 'Professional' in f]
                other_html = [f for f in all_files if f.endswith('.html') and 'Professional' not in f] if not prof_html else []
                html_files = prof_html + other_html
                # Include only Professional PDF reports
                prof_pdf = [f for f in all_files if f.endswith('.pdf') and 'Professional_Equity_Report' in f]
                other_pdf = [f for f in all_files if f.endswith('.pdf') and 'Equity_Report' in f and f not in prof_pdf] if not prof_pdf else []
                pdf_files = prof_pdf + other_pdf
            result.append({
                "task_id": r.task_id,
                "ticker": r.ticker,
                "company_name": r.company_name,
                "status": r.status or "completed",
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "html": html_files,
                "pdf": pdf_files,
            })
        db.close()
        return result
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")
        return []


@app.delete("/api/history/{task_id}")
async def delete_history(task_id: str, request: Request):
    """Delete the report request record (and all records for the same ticker)."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        db = SessionLocal()
        # Look up the ticker for this task_id
        report = db.query(ReportRequest).filter(
            ReportRequest.task_id == task_id,
            ReportRequest.user_id == user["id"]
        ).first()
        if not report:
            db.close()
            raise HTTPException(status_code=404, detail="Report not found")
        ticker = report.ticker
        # Remove all records for this user+ticker combination
        db.query(ReportRequest).filter(
            ReportRequest.ticker == ticker,
            ReportRequest.user_id == user["id"]
        ).delete()
        db.commit()
        db.close()
        # Also evict from in-memory store
        if task_id in tasks:
            del tasks[task_id]
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to delete report: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete report")


@app.get("/api/reports/{ticker}")
async def list_reports(ticker: str, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    report_dir = os.path.join(OUTPUT_DIR, ticker, "report")
    if not os.path.exists(report_dir):
        return {"reports": []}
    
    reports = [f for f in os.listdir(report_dir) if f.endswith((".html", ".pdf"))]
    return {"reports": reports}