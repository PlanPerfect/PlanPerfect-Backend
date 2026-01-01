from fastapi import APIRouter
from fastapi.responses import JSONResponse
from Services import DatabaseManager as DM
from Services import Utilities

router = APIRouter(prefix="/database", tags=["Tools & Services"])

@router.get("/data")
def get_data(path: str = ""):
    try:
        if not path:
            return DM.data

        path_parts = path.split("/")

        value = DM.peek(path_parts)

        if value is None:
            return JSONResponse(status_code=404, content={ "success": False, "result": "ERROR: Data does not exist." })

        return JSONResponse(status_code=200, content={ "success": True, "result": value })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to get data; {e}" })

@router.post("/data")
def set_data(path: str):
    try:
        path_parts = path.split("/")

        sample_data = {
            "ID": Utilities.GenerateRandomID(12),
            "value": Utilities.GenerateRandomInt(1, 100)
        }

        target = DM.data
        for part in path_parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[path_parts[-1]] = sample_data

        DM.save()

        return JSONResponse(status_code=200, content={ "success": True, "result": f"SUCCESS: Data set successfully. Path: {path}" })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to set data. Error: {e}" })

@router.delete("/data")
def delete_data(path: str):
    try:
        path_parts = path.split("/")

        destroy = DM.destroy(path_parts)

        if destroy:
            DM.save()

            return JSONResponse(status_code=200, content={ "success": True, "result": f"SUCCESS: Data deleted successfully. Path: {path}" })
        else:
            return JSONResponse(status_code=404, content={ "success": False, "result": "ERROR: Data does not exist." })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to delete data. Error: {e}" })