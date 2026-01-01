import os
import sys
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin.exceptions import FirebaseError

class Bootcheck:
    REQUIRED_ENV_VARS = [ # IMPORTANT: add all .env variable keys here
        "PORT",
        "DEV_MODE",
        "SMTP_HOST",
        "SMTP_PORT",
        "EMAIL_SENDER",
        "SMTP_APP_PASSWORD",
        "FIREBASE_DATABASE_URL",
        "FIREBASE_CREDENTIALS_PATH",
    ]

    @staticmethod
    def check_env_file():
        if not os.path.exists('.env'):
            print("BOOTCHECK - ERROR: Environment file not found. Please create a .env file with the required environment variables.\n")
            return False

        return True

    @staticmethod
    def check_env_variables():
        missing_vars = []

        for var in Bootcheck.REQUIRED_ENV_VARS:
            if var not in os.environ or os.environ[var] == '':
                missing_vars.append(var)

        if missing_vars:
            print(f"BOOTCHECK - ERROR: Missing environment variables: {', '.join(missing_vars)}.\n")
            return False

        return True

    @staticmethod
    def check_firebase_key_file():
        creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

        if not os.path.exists(creds_path):
            print(f"BOOTCHECK - ERROR: Firebase Service Account Key file not found.\n")
            return False

        return True

    @staticmethod
    def validate_firebase_credentials():
        database_url = os.getenv("FIREBASE_DATABASE_URL")
        creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

        try:
            with open(creds_path, 'r') as f:
                json_content = json.load(f)

            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in json_content]

            if missing_fields:
                print(f"BOOTCHECK - ERROR: Firebase Service Account Key file is missing required fields: {', '.join(missing_fields)}.\n")
                return False

            if not firebase_admin._apps:
                cred = credentials.Certificate(creds_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })

            test_ref = db.reference('/test_connection')
            test_ref.set({'test': 'connection_check', 'timestamp': datetime.now().isoformat()})
            test_ref.delete()

            apps = firebase_admin._apps.copy()
            for app_name, app in apps.items():
                try:
                    firebase_admin.delete_app(app)
                except:
                    pass

            return True

        except json.JSONDecodeError as e:
            print(f"BOOTCHECK - ERROR: Firebase Service Account Key file is corrupted. Error: {str(e)}.\n")
            return False

        except (FirebaseError, ValueError, IOError) as e:
            print(f"BOOTCHECK - ERROR: Firebase connection test failed. Error: {str(e)}.\n")
            return False

    @staticmethod
    def set_python_dont_write_bytecode():
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        return True

    @staticmethod
    def run_checks():
        checks = [
            ("Environment file check", Bootcheck.check_env_file),
            ("Environment variables check", Bootcheck.check_env_variables),
            ("Firebase key file check", Bootcheck.check_firebase_key_file),
            ("Firebase credentials validation", Bootcheck.validate_firebase_credentials),
            ("Python bytecode setting", Bootcheck.set_python_dont_write_bytecode)
        ]

        for check_name, check_func in checks:
            if not check_func():
                print(f"BOOTCHECK - ERROR: Check failed at: {check_name}. Startup aborted.\n")
                return False

        print("BOOTCHECK COMPLETE, SYSTEM READY. SERVER STARTING...\n")
        return True


if __name__ == "__main__":
    if Bootcheck.run_checks():
        sys.exit(0)
    else:
        sys.exit(1)