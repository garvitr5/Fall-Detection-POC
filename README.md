# Fall Detection

A Streamlit interface for classifying still images with the trained ResNet50
fall-detection model.

## Run locally

```powershell
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including
   `streamlit_app/model/final_resnet50.keras`.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file path to `streamlit_app/app.py`.
4. Choose Python 3.12 in Advanced settings, then deploy.

The app directory contains its own `requirements.txt`, which Community Cloud
uses during the build.
