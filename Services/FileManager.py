import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from typing import Dict, Optional, Union

from fastapi import UploadFile
from Services import Logger
import os

"""
FileManager handles Cloudinary image uploads and optimized URL generation.
Implements a singleton pattern for consistent cloud access.
"""

class FileManagerClass:
    _instance = None

    SUPPORTED_TYPES = {'png', 'jpg', 'jpeg', 'webp', 'avif', 'gif', 'svg'}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        os.environ['CLOUDINARY_URL'] = os.getenv('CLOUDINARY_URL')
        cloudinary.config(secure=True)

        self._folder_name = "PlanPerfect Cloud Storage"
        self._initialized = True

        print(
            f"FILE MANAGER INITIALISED. CONNECTED TO CLOUD: \033[94m{cloudinary.config().cloud_name}\033[0m, "
            f"FOLDER: \033[94m{self._folder_name}\033[0m\n"
        )

    def _is_supported_image(self, filename: str) -> bool:
        extension = filename.split('.')[-1].lower()
        return extension in self.SUPPORTED_TYPES

    def store_file(
        self,
        file: Union[bytes, UploadFile],
        subfolder: Optional[str] = None,
    ) -> Dict[str, str]:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            if hasattr(file, "filename"):
                filename = file.filename
                file_content = file.file.read()
            else:
                raise ValueError("Expected an UploadFile instance as 'file'.")

            if not self._is_supported_image(filename):
                raise ValueError(f"Unsupported file type for {filename}.")

            folder_path = f"{self._folder_name}/{subfolder}" if subfolder else self._folder_name
            public_id = filename.rsplit('.', 1)[0]

            upload_result = cloudinary.uploader.upload(
                file_content,
                folder=folder_path,
                public_id=public_id,
                resource_type="image"
            )

            file_id = upload_result['public_id']
            optimized_url = self.get_optimized_url(file_id)

            return {"file_id": file_id, "url": optimized_url}

        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to store file. Error: {str(e)}")
            raise

    def delete_file(self, file_id: str) -> bool:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            result = cloudinary.uploader.destroy(file_id, resource_type="image")
            return result.get('result') == 'ok'
        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to delete file {file_id}. Error: {str(e)}")
            return False

    def get_optimized_url(self, file_id: str) -> str:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            url, _ = cloudinary_url(
                file_id,
                fetch_format="auto",
                quality="auto",
                secure=True
            )
            return url
        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to generate optimized URL. Error: {str(e)}")
            raise


FileManager = FileManagerClass()