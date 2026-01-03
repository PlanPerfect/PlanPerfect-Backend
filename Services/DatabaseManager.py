import os
import copy
import firebase_admin
from firebase_admin import credentials, db
from typing import Optional, Any
from Services import Logger

class AutoDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, AutoDict):
            value = AutoDict(value)
            self[key] = value
        return value

    def __deepcopy__(self, memo):
        return AutoDict(copy.deepcopy(dict(self), memo))


class DatabaseManagerClass:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._db_ref = None
            cls._instance._app = None
            cls._instance.data = AutoDict()
            cls._instance._listener = None
            cls._instance._cloudsync_enabled = False
            cls._instance._first_sync = True
        return cls._instance

    def initialize(
        self,
        database_url: str,
        credentials_path: Optional[str] = None,
        credentials_dict: Optional[dict] = None
    ) -> None:
        if self._initialized:
            return

        try:
            if credentials_path:
                cred = credentials.Certificate(credentials_path)
            elif credentials_dict:
                cred = credentials.Certificate(credentials_dict)
            else:
                Logger.log("[DATABASE MANAGER] - ERROR: Credentials not provided.")
                raise ValueError("[DATABASE MANAGER] - ERROR: Credentials not provided.")

            self._app = firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })

            self._db_ref = db.reference()
            self._initialized = True
            self._cloudsync_enabled = os.getenv('CLOUDSYNC_ENABLED').lower() == 'true'
            print(f"DATABASE MANAGER INITIALISED. CONNECTED TO \033[94m{database_url}\033[0m\n")

            self.load()

            if self._cloudsync_enabled:
                self._start_listener()

        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: DatabaseManager could not be initialized. Error: {e}")
            raise

    def load(self) -> dict:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        try:
            snapshot = self._db_ref.get()
            if snapshot is not None:
                self.data = self._convert_to_autodict(snapshot)
            else:
                self.data = AutoDict()

            sync_status = "ENABLED" if self._cloudsync_enabled else "DISABLED"
            print(f"CloudSync {sync_status}. SUCCESSFULLY LOADED CLOUD FIREBASE RTDB INTO LOCAL MEMORY.\n")

            return self.data
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to load Firebase RTDB. Error: {e}")
            raise

    def _start_listener(self) -> None:
        def listener(event):
            if self._first_sync:
                self._first_sync = False
                return
            try:
                snapshot = self._db_ref.get()
                if snapshot is not None:
                    self.data = self._convert_to_autodict(snapshot)
                else:
                    self.data = AutoDict()
                print("[DATABASE MANAGER] - LIVE TAIL: Changes detected in RTDB Console. In-memory copy synced.")
            except Exception as e:
                Logger.log(f"[DATABASE MANAGER] - ERROR: CloudSync listener failed. Error: {e}")

        self._listener = self._db_ref.listen(listener)

    def _convert_to_autodict(self, data: Any) -> Any:
        if isinstance(data, dict):
            result = AutoDict()
            for key, value in data.items():
                result[key] = self._convert_to_autodict(value)
            return result
        elif isinstance(data, list):
            return [self._convert_to_autodict(item) for item in data]
        else:
            return data

    def save(self) -> bool:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        try:
            regular_dict = self._convert_to_regular_dict(self.data)
            self._db_ref.set(regular_dict)
            return True
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to save to Firebase RTDB. Error: {e}")
            return False

    def _convert_to_regular_dict(self, data: Any) -> Any:
        if isinstance(data, AutoDict):
            result = {}
            for key, value in data.items():
                result[key] = self._convert_to_regular_dict(value)
            return result
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._convert_to_regular_dict(value)
            return result
        elif isinstance(data, list):
            return [self._convert_to_regular_dict(item) for item in data]
        else:
            return data

    def peek(self, path: list) -> Any:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        try:
            value = self.data
            for key in path:
                if isinstance(value, (dict, AutoDict)) and key in value:
                    value = value[key]
                else:
                    return None
            return copy.deepcopy(value)
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to peek at path {path}. Error: {e}")
            return None

    def set_value(self, path: list, value: Any) -> bool:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        if not path:
            Logger.log("[DATABASE MANAGER] - ERROR: Path cannot be empty.")
            return False

        try:
            current = self.data
            for key in path[:-1]:
                if key not in current or not isinstance(current[key], (dict, AutoDict)):
                    current[key] = AutoDict()
                current = current[key]
            current[path[-1]] = value
            return True
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to set value at path {path}. Error: {e}")
            return False

    def destroy(self, path: list) -> bool:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        if not path or len(path) == 0:
            Logger.log("[DATABASE MANAGER] - ERROR: Cannot delete root of database.")
            return False

        try:
            current = self.data
            for key in path[:-1]:
                if isinstance(current, (dict, AutoDict)) and key in current:
                    current = current[key]
                else:
                    Logger.log(f"[DATABASE MANAGER] - ERROR: Path {path} does not exist.")
                    return False

            final_key = path[-1]
            if isinstance(current, (dict, AutoDict)) and final_key in current:
                del current[final_key]
                return True
            else:
                Logger.log(f"[DATABASE MANAGER] - ERROR: Key {final_key} not found.")
                return False

        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to delete path {path}. Error: {e}")
            return False

    def reload(self) -> dict:
        return self.load()

    def stop_cloudsync(self) -> None:
        if self._listener:
            self._listener.close()
            self._listener = None


DatabaseManager = DatabaseManagerClass()