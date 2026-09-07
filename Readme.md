## Clone repository
```bash
git clone https://github.com/zulee1711/cc_fraud_detection_using_QRC.git
```
**Note:** To change the name of the cloned folder, use the following command:
```bash
git clone https://github.com/zulee1711/cc_fraud_detection_using_QRC.git <new_folder_name>
```
with `<new_folder_name>` being the desired name for the cloned folder.

## Download datasets
Datasets are available at:
[Credit Risk Datasets](https://tud365.sharepoint.com/:f:/r/sites/stud-JIPQuobly/Gedeelde%20documenten/Datasets?d=w89895bce21bd4788ad9c63fe4314ead9&csf=1&web=1&e=KTtJkA).

Download the datasets and place them in the `raw_data` folder of the cloned repository.

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

## Run examples:
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