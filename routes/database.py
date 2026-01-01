from fastapi import APIRouter
from fastapi.responses import JSONResponse
from Services import DatabaseManager as DM
from Services import Utilities

router = APIRouter(prefix="/database", tags=["Tools & Services"])

@router.get("/data")
def get_data():
    try:
        value = DM.peek([])

        if value is None:
            return JSONResponse(status_code=404, content={ "success": False, "result": "ERROR: Data does not exist." })

        return JSONResponse(status_code=200, content={ "success": True, "result": value })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to get data; {e}" })

@router.post("/data")
def set_data():
    try:
        ID = Utilities.GenerateRandomID(12)

        sample_data = {
            "ID": ID,
            "value": Utilities.GenerateRandomInt(1, 100)
        }

        DM.data["SAMPLE_1"]["TEST"]["BRANCH"][ID] = sample_data

        DM.save()

        return JSONResponse(status_code=200, content={ "success": True, "result": f"SUCCESS: Data set successfully." })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to set data. Error: {e}" })

@router.delete("/data")
def delete_data():
    try:
        destroy = DM.destroy([]) # destroys the root. you could also do something like DM.destroy(["SAMPLE_1", "TEST", "BRANCH", "some_id"])

        if destroy:
            DM.save()

            return JSONResponse(status_code=200, content={ "success": True, "result": f"SUCCESS: Data deleted successfully." })
        else:
            return JSONResponse(status_code=404, content={ "success": False, "result": "ERROR: Data does not exist." })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to delete data. Error: {e}" })