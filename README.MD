
 >> for running the project

 venv\Scripts\activate  
 python app.py 







1: python -m venv venv  ----install virtual env

3: python -m pip install --upgra


de pip    -----upgrade pip
   pip install scipy   --- insall scipy( dependancy)
   pip install pillow numpy    --- install dependancy
   pip install tensorflow-cpu==2.12.0   ---- dependancy
 
4: python train_model.py   -- train data model 




**-------------------------------------------------------------------------------------------**

# 🐶🐱 Cat vs Dog Image Classifier (CNN + Python)

A deep learning project using TensorFlow CNN to classify images as Cat or Dog, UI.  
Supports multiple image upload, smooth UI, and Python 3.10.11 compatible.

---

## 🔧 Version

- Python **3.10.11** 

## 📁 PROJECT STRUCTURE


CatsDogsCNN/
│
├── dataset/ ----- dataset 
│ ├── train/
│ │ ├── cats/
│ │ └── dogs/
│ └── test/
│ ├── cats/
│ └── dogs/
│
├── model/     ---- datamodel generated after running train_model.py
│ └── cnn_model.h5
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md


---

## 🧠 STEP 1: CREATE VIRTUAL ENVIRONMENT

Open VS Code Terminal in project folder:

```bash
python -m venv venv
venv\Scripts\activate
📦 STEP 2: INSTALL DEPENDENCIES
pip install -r requirements.txt

If scipy error aaye:

pip install scipy
🐕🐈 STEP 3: DATASET SETUP

Dataset structure MUST BE EXACT:

dataset/train/cats/
dataset/train/dogs/
dataset/test/cats/
dataset/test/dogs/

Images:

Cats → cat images

Dogs → dog images

🧪 STEP 4: TRAIN THE MODEL (ONE TIME)
python train_model.py

✔ Model save ho jayega:

model/cnn_model.h5
🎨 STEP 5: RUN UI APPLICATION
python app.py

python --version
# Should be 3.10.x


** RUN COMMAND SUMMARY **

venv\Scripts\activate
python train_model.py
python app.py
© CREDITS

UI Base: Kritika
Model & Improvements: AI Assisted
Free for learning & academic use.

**🚀 FUTURE SCOPE  **

Webcam live detection

Convert to .exe

Web app (Streamlit)

Accuracy improvement (95%+)