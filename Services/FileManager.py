import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from typing import Dict, Optional, Union, Literal
import requests

from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from Services import Logger
import os
import io

"""
FileManager handles Cloudinary file uploads and optimized URL generation.
Supports images and PDFs with intelligent optimization.
Implements a singleton pattern for consistent cloud access.
"""

class FileManagerClass:
    _instance = None

    SUPPORTED_IMAGE_TYPES = {'png', 'jpg', 'jpeg', 'webp', 'avif', 'gif', 'svg'}
    SUPPORTED_DOCUMENT_TYPES = {'pdf'}
    SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_DOCUMENT_TYPES

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

    def _get_file_extension(self, filename: str) -> str:
        return filename.split('.')[-1].lower()

    def _is_supported_file(self, filename: str) -> bool:
        extension = self._get_file_extension(filename)
        return extension in self.SUPPORTED_TYPES

    def _get_resource_type(self, filename: str) -> Literal["image", "raw"]:
        extension = self._get_file_extension(filename)
        if extension in self.SUPPORTED_IMAGE_TYPES:
            return "image"
        elif extension in self.SUPPORTED_DOCUMENT_TYPES:
            return "raw"
        else:
            raise ValueError(f"Unsupported file type: {extension}")

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

            if not self._is_supported_file(filename):
                raise ValueError(
                    f"Unsupported file type for {filename}. "
                    f"Supported types: {', '.join(self.SUPPORTED_TYPES)}"
                )

            folder_path = f"{self._folder_name}/{subfolder}" if subfolder else self._folder_name
            public_id = filename.rsplit('.', 1)[0]
            resource_type = self._get_resource_type(filename)

            upload_result = cloudinary.uploader.upload(
                file_content,
                folder=folder_path,
                public_id=public_id,
                resource_type=resource_type
            )

            file_id = upload_result['public_id']

            if resource_type == "image":
                url = self.get_optimized_url(file_id)
            else:
                url = self.get_pdf_url(file_id)

            return {
                "file_id": file_id,
                "url": url
            }

        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to store file. Error: {str(e)}")
            raise

    def delete_file(self, file_id: str, resource_type: str = "image") -> bool:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            result = cloudinary.uploader.destroy(file_id, resource_type=resource_type)
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

    def get_pdf_url(self, file_id: str) -> str:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            url, _ = cloudinary_url(
                file_id,
                resource_type="raw",
                secure=True
            )
            return url
        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to generate PDF URL. Error: {str(e)}")
            raise

    def get_pdf_file(self, file_id: str) -> StreamingResponse:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            pdf_url = self.get_pdf_url(file_id)

            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            filename = file_id.split('/')[-1] + '.pdf'

            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'inline; filename="{filename}"',
                    "Content-Type": "application/pdf"
                }
            )
        except requests.exceptions.RequestException as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to fetch PDF from Cloudinary. Error: {str(e)}")
            raise
        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to retrieve PDF file. Error: {str(e)}")
            raise

    def get_pdf_thumbnail(
        self,
        file_id: str,
        page: int = 1,
        width: Optional[int] = 300,
        height: Optional[int] = None
    ) -> str:
        if not self._initialized:
            raise RuntimeError("FileManager not initialized. Call initialize() first.")

        try:
            transformation = {
                'page': page,
                'format': 'jpg',
                'quality': 'auto',
                'secure': True
            }

            if width:
                transformation['width'] = width
            if height:
                transformation['height'] = height
            if width and height:
                transformation['crop'] = 'fill'

            url, _ = cloudinary_url(file_id, **transformation)
            return url
        except Exception as e:
            Logger.log(f"[FILE MANAGER] - ERROR: Failed to generate PDF thumbnail. Error: {str(e)}")
            raise

    def get_pdf_preview_urls(
        self,
        file_id: str,
        num_pages: int = 3,
        width: int = 300
    ) -> list[str]:
        return [
            self.get_pdf_thumbnail(file_id, page=page, width=width)
            for page in range(1, num_pages + 1)
        ]


FileManager = FileManagerClass()