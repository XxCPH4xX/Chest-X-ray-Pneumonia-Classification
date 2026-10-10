import os

from backend.app.config import Settings
from backend.app.main import create_app

hosts = {
    value.strip()
    for value in (
        os.getenv("VERCEL_URL", ""),
        os.getenv("VERCEL_PROJECT_PRODUCTION_URL", ""),
        os.getenv("VERCEL_BRANCH_URL", ""),
        *os.getenv("ALLOWED_HOSTS", "").split(","),
    )
    if value.strip()
}
if not hosts:
    raise RuntimeError("Set VERCEL_URL or ALLOWED_HOSTS to the deployment hostname")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/chest-xray-matplotlib")
app = create_app(
    Settings(
        allowed_hosts=sorted(hosts),
        origins=[f"https://{host}" for host in sorted(hosts)],
        history_enabled=False,
        max_upload_bytes=3 * 1024 * 1024,
    )
)
