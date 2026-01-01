import firebase_admin
from firebase_admin import credentials, db
from typing import Optional
from Services import Logger
import copy


class DatabaseManagerClass:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._db_ref = None
            cls._instance._app = None
            cls._instance.data = {}
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
            print(f"DATABASE MANAGER INITIALISED. CONNECTED TO \033[94m{database_url}\033[0m\n")

            self.load()

        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: DatabaseManager could not be initialized. Error: {e}")
            raise

    def load(self) -> dict:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        try:
            snapshot = self._db_ref.get()
            self.data = snapshot if snapshot is not None else {}
            print("SUCCESSFULLY LOADED FIREBASE RTDB INTO LOCAL MEMORY.\n")
            return self.data
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to load FIREBASE RTDB. Error: {e}")
            raise

    def save(self) -> bool:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        try:
            self._db_ref.set(self.data)
            return True
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to save to FIREBASE RTDB. Error: {e}")
            return False

    def peek(self, path: list) -> any:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        try:
            value = self.data
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return copy.deepcopy(value)
        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to peek at path {path}. Error: {e}")
            return None

    def destroy(self, path: list) -> bool:
        if not self._initialized:
            Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
            raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")

        if not path or len(path) == 0:
            Logger.log("[DATABASE MANAGER] - ERROR: Cannot delete root of database.")
            return False

        try:
            parent = self.data
            for key in path[:-1]:
                if isinstance(parent, dict) and key in parent:
                    parent = parent[key]
                else:
                    Logger.log(f"[DATABASE MANAGER] - ERROR: Path {path} does not exist.")
                    return False

            final_key = path[-1]
            if isinstance(parent, dict) and final_key in parent:
                del parent[final_key]
                return True
            else:
                Logger.log(f"[DATABASE MANAGER] - ERROR: Key {final_key} not found.")
                return False

        except Exception as e:
            Logger.log(f"[DATABASE MANAGER] - ERROR: Failed to delete path {path}. Error: {e}")
            return False

    def reload(self) -> dict:
        return self.load()


DatabaseManager = DatabaseManagerClass()