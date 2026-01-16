# Getting started

### Setting up environment
- Create a new virtual environment
  - `python3 -m venv planperfect-env`
- Activate virtual environment
  - Windows: `planperfect-env\Scripts\activate`
  - MacOS: `source planperfect-env/bin/activate`
- Install Dependancies
  - `pip install -r requirements.txt`
  - `pip install -r requirements-sd.txt`
  - `pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu121`
- Running the server
  - `python app.py` or `python3 app.py`
