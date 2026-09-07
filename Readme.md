## Setup
### Check Python version:
`python --version`
Check if the version is 3.11 or higher. If not, install the latest version of Python from the official website: https://www.python.org/downloads/

### Create a virtual environment and activate it:
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the application:
You can run the Jupyter Notebook or the Python script to see the basic usage of the application.
To run the Python script, use the following commands:
On Windows:
```bash
cd examples
python BasicUsage.py
```

On macOS/Linux:
```bash
cd examples
python3 BasicUsage.py
```