import datetime

LOG_FILE = 'app.log'

"""
    Logger is a service which helps log key system messages to the app log, which is viewable at "/sample/logs-page" on system startup.
    It is a key service during the development phase for capturing crucial system messages and failures.
"""

def log(message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"

    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)