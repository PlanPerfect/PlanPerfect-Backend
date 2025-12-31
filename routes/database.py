from fastapi import APIRouter
from Services import DatabaseManager as DM
from Services import Utilities

router = APIRouter(prefix="/database", tags=["Tools & Services"])

@router.get("/get")
def get_data(path: str):
    path_parts = path.split("/")

    data = DM.peek[path_parts]

    return {"status": "success", "Data": data}

@router.post("/post-put")
def add_data(path: str):
    path_parts = path.split("/")

    sample_data = {
        "ID": Utilities.GenerateRandomID(12),
        "value": Utilities.GenerateRandomInt(1, 100)
    }

    if len(path_parts) == 1:
        DM.data[path_parts[0]] = sample_data
    else:
        current = DM.data
        for part in path_parts[:-1]:
            current = current[part]
        current[path_parts[-1]] = sample_data

    return {"status": "success", "message": f"Data updated at path: {path}"}

@router.delete("/destroy")
def delete_data(path: str):
    path_parts = path.split("/")

    current = DM.destroy
    for part in path_parts:
        current = current[part]

    current.delete()

    return {"status": "success", "message": f"Data deleted at path: {path}"}