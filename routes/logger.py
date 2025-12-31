from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from Services import Logger
import re

router = APIRouter(prefix="/logs", tags=["Tools & Services"])

LOG_FILE = 'app.log'

LOG_LINE_RE = re.compile(r'^\[(?P<timestamp>[\d\- :]+)\]\s+\[(?P<tag>[^\]]+)\]\s*-\s*(?P<message>.*)$')

def read_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        logs = []
        for line in lines:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            m = LOG_LINE_RE.match(line)
            if not m:
                logs.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "tag": "UNKNOWN",
                    "raw_message": line,
                    "level": "MISC",
                    "display_message": line,
                })
                continue

            ts = m.group('timestamp')
            tag = m.group('tag')
            message = m.group('message').strip()

            level = 'MISC'
            for prefix in ('SUCCESS:', 'UERROR:', 'ERROR:'):
                if message.upper().startswith(prefix):
                    level = prefix[:-1]  # remove trailing ':'
                    message = message[len(prefix):].strip()
                    break

            display_message = f"[{tag}] - {message}"

            logs.append({
                "timestamp": ts,
                "tag": tag,
                "raw_message": m.group('message'),
                "level": level,
                "display_message": display_message,
            })

        return logs[::-1]
    except FileNotFoundError:
        return []

@router.post("/log-message")
def log_message(message: str):
    Logger.log(message)
    return {"status": "Message logged"}

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def logs_page(request: Request):
    logs = read_logs()

    log_levels = {
        'SUCCESS': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        'UERROR': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'ERROR': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
        'MISC': 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
    }

    def get_log_level_from_entry(entry):
        return entry.get('level', 'MISC')

    log_cards_html = ""
    if not logs:
        log_cards_html = """
        <div class="empty-state">
            <div class="empty-icon">📝</div>
            <h3>No Logs Found</h3>
            <p>Application logs will appear here once events are logged.</p>
        </div>
        """
    else:
        for log_entry in logs:
            level = get_log_level_from_entry(log_entry)
            gradient = log_levels.get(level, log_levels['MISC'])

            try:
                dt = datetime.strptime(log_entry['timestamp'], '%Y-%m-%d %H:%M:%S')
                formatted_time = dt.strftime('%I:%M %p')
                formatted_date = dt.strftime('%b %d, %Y')
            except Exception:
                formatted_time = log_entry['timestamp']
                formatted_date = ''

            safe_message_attr = log_entry['display_message'].replace("'", "&#39;").replace('"', "&quot;").replace('\n', ' ')

            display_message_html = log_entry['display_message'].replace('<', '&lt;').replace('>', '&gt;')

            log_cards_html += f"""
            <div class="log-card" data-level="{level}">
                <div class="log-card-header">
                    <div class="log-level-badge" style="background: {gradient};">
                        {level}
                    </div>
                    <div class="log-time">
                        <span class="log-time-hour">{formatted_time}</span>
                        <span class="log-time-date">{formatted_date}</span>
                    </div>
                </div>
                <div class="log-message">
                    <p>{display_message_html}</p>
                </div>
                <div class="log-footer">
                    <span class="log-timestamp">
                        {log_entry['timestamp']}
                    </span>
                    <div class="log-actions">
                        <button class="copy-log-btn" data-message="{safe_message_attr}" title="Copy message">
                            <i>📋</i>
                        </button>
                    </div>
                </div>
            </div>
            """

    level_counts = {'SUCCESS': 0, 'UERROR': 0, 'ERROR': 0, 'MISC': 0}
    for log_entry in logs:
        lvl = get_log_level_from_entry(log_entry)
        if lvl in level_counts:
            level_counts[lvl] += 1
        else:
            level_counts['MISC'] += 1

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Application Logs - PlanPerfect</title>
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
                    <a href="/logs" class="nav-link active">Logs</a>
                </div>
            </div>
        </nav>

        <main class="logs-container">
            <div class="logs-header">
                <h1 class="logs-title">Application Logs</h1>

                <div class="stats-overview">
                    <div class="stat-card">
                        <div class="stat-value" style="color: #10b981;">{level_counts['SUCCESS']}</div>
                        <div class="stat-label">Success</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #f59e0b;">{level_counts['UERROR']}</div>
                        <div class="stat-label">User Errors</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #ef4444;">{level_counts['ERROR']}</div>
                        <div class="stat-label">System Errors</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #8b5cf6;">{level_counts['MISC']}</div>
                        <div class="stat-label">Miscellaneous</div>
                    </div>
                </div>
            </div>

            <div class="logs-controls">
                <div class="log-filters">
                    <button class="filter-btn active" onclick="filterLogs('ALL')">All</button>
                    <button class="filter-btn" onclick="filterLogs('SUCCESS')">Success</button>
                    <button class="filter-btn" onclick="filterLogs('UERROR')">User Errors</button>
                    <button class="filter-btn" onclick="filterLogs('ERROR')">System Errors</button>
                    <button class="filter-btn" onclick="filterLogs('MISC')">Miscellaneous</button>
                </div>
            </div>

            <div class="logs-grid" id="logsGrid">
                {log_cards_html}
                {"<div class='no-logs-message' id='noLogsMessage'>No logs found for the selected filter.</div>" if logs else ""}
            </div>
        </main>

        <footer class="footer">
            <p>PlanPerfect Backend Log Viewer • Showing {len(logs)} log entries</p>
        </footer>

        <script>
            function filterLogs(level) {{
                // Update active filter button
                document.querySelectorAll('.filter-btn').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                if (event && event.target) {{
                    event.target.classList.add('active');
                }}

                const logCards = document.querySelectorAll('.log-card');
                const noLogsMessage = document.getElementById('noLogsMessage');
                let visibleCount = 0;

                logCards.forEach(card => {{
                    card.classList.remove('hidden');
                    if (level === 'ALL' || card.getAttribute('data-level') === level) {{
                        visibleCount++;
                    }} else {{
                        card.classList.add('hidden');
                    }}
                }});

                if (visibleCount === 0) {{
                    if (noLogsMessage) noLogsMessage.style.display = 'block';
                }} else {{
                    if (noLogsMessage) noLogsMessage.style.display = 'none';
                }}
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                // Add animation delays to log cards
                const logCards = document.querySelectorAll('.log-card');
                logCards.forEach((card, index) => {{
                    card.style.setProperty('--card-index', index);
                    card.style.animationDelay = `${{index * 0.05}}s`;
                }});

                // Add event listeners to all copy buttons
                document.querySelectorAll('.copy-log-btn').forEach(btn => {{
                    btn.addEventListener('click', function() {{
                        const message = this.getAttribute('data-message')
                            .replace(/&#39;/g, "'")
                            .replace(/&quot;/g, '"');

                        navigator.clipboard.writeText(message).then(() => {{
                            const originalHTML = this.innerHTML;
                            const originalClass = this.className;
                            this.innerHTML = '✓ Copied to clipboard';
                            this.className = 'copy-log-btn copied';

                            setTimeout(() => {{
                                this.innerHTML = originalHTML;
                                this.className = originalClass;
                            }}, 2000);
                        }}).catch(err => {{
                            console.error('Failed to copy: ', err);
                        }});
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)