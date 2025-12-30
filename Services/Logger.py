import datetime

LOG_FILE = 'app.log'

def log(message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"

    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)

    print(log_entry.strip())