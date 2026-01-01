# import firebase_admin
# from firebase_admin import credentials, db
# from typing import Any, List, Optional
# from Services import Logger

# class DatabasePath:
#     def __init__(self, manager: 'DatabaseManagerClass', path: List[str] = None):
#         self._manager = manager
#         self._path = path or []
#         self._value = None
#         self._fetched = False

#     def __getitem__(self, key: str) -> 'DatabasePath':
#         new_path = DatabasePath(self._manager, self._path + [key])
#         path_str = '/'.join(map(str, new_path._path))
#         ref = self._manager._db_ref.child(path_str)
#         new_path._value = ref.get()
#         new_path._fetched = True
#         return new_path

#     def __setitem__(self, key: str, value: Any):
#         path = self._path + [key]
#         path_str = '/'.join(map(str, path))
#         ref = self._manager._db_ref.child(path_str)
#         ref.set(value)

#     def __repr__(self):
#         if self._fetched:
#             return repr(self._value)
#         return super().__repr__()

#     def __str__(self):
#         if self._fetched:
#             return str(self._value)
#         return super().__str__()

#     def __bool__(self):
#         if self._fetched:
#             return bool(self._value)
#         return True

#     def __eq__(self, other):
#         if self._fetched:
#             return self._value == other
#         return super().__eq__(other)

#     def __iter__(self):
#         if self._fetched and self._value is not None:
#             return iter(self._value)
#         return iter([])

#     @property
#     def value(self):
#         return self._value


# class PeekPath:
#     def __init__(self, manager: 'DatabaseManagerClass'):
#         self._manager = manager

#     def __getitem__(self, key: str) -> Any:
#         if isinstance(key, str):
#             path = [key]
#         else:
#             path = list(key) if hasattr(key, '__iter__') else [key]

#         path_str = '/'.join(map(str, path))
#         ref = self._manager._db_ref.child(path_str)
#         return ref.get()


# class DestroyPath:
#     def __init__(self, manager: 'DatabaseManagerClass', path: List[str] = None):
#         self._manager = manager
#         self._path = path or []

#     def __getitem__(self, key: str) -> 'DestroyPath':
#         return DestroyPath(self._manager, self._path + [key])

#     def delete(self):
#         if not self._path:
#             Logger.log("[DATABASE MANAGER] - ERROR: Cannot delete root of database.")
#             return

#         path_str = '/'.join(map(str, self._path))
#         ref = self._manager._db_ref.child(path_str)

#         if len(self._path) > 1:
#             parent_path = '/'.join(map(str, self._path[:-1]))
#             parent_ref = self._manager._db_ref.child(parent_path)
#             parent_data = parent_ref.get()

#             if parent_data and isinstance(parent_data, dict):
#                 if len(parent_data) == 1 and self._path[-1] in parent_data:
#                     parent_ref.child(self._path[-1]).delete()
#                 else:
#                     ref.delete()
#             else:
#                 ref.delete()
#         else:
#             ref.delete()

# class DatabaseManagerClass:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialized = False
#             cls._instance._db_ref = None
#             cls._instance._app = None
#         return cls._instance

#     def initialize(
#         self,
#         database_url: str,
#         credentials_path: Optional[str] = None,
#         credentials_dict: Optional[dict] = None
#     ) -> None:
#         if self._initialized:
#             return

#         try:
#             if credentials_path:
#                 cred = credentials.Certificate(credentials_path)
#             elif credentials_dict:
#                 cred = credentials.Certificate(credentials_dict)
#             else:
#                 Logger.log("[DATABASE MANAGER] - ERROR: Credentials not provided.")
#                 raise ValueError("[DATABASE MANAGER] - ERROR: Credentials not provided.")

#             self._app = firebase_admin.initialize_app(cred, {
#                 'databaseURL': database_url
#             })

#             self._db_ref = db.reference()
#             self._initialized = True
#             print(f"DATABASE MANAGER INITIALISED. CONNECTED TO \033[94m{database_url}\033[0m\n")

#         except Exception as e:
#             Logger.log(f"[DATABASE MANAGER] - ERROR: DatabaseManager could not be initialized. Error: {e}")
#             raise

#     @property
#     def data(self) -> DatabasePath:
#         if not self._initialized:
#             Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
#             raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
#         return DatabasePath(self)

#     @property
#     def peek(self) -> PeekPath:
#         if not self._initialized:
#             Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
#             raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
#         return PeekPath(self)

#     @property
#     def destroy(self) -> DestroyPath:
#         if not self._initialized:
#             Logger.log("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
#             raise RuntimeError("[DATABASE MANAGER] - ERROR: DatabaseManager not initialized. Call DM.initialize() first.")
#         return DestroyPath(self)


# DatabaseManager = DatabaseManagerClass()

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