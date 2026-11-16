Copy app.py, flood_service.py and requirements.txt into the SAME folder as urban_flood.py
(overwrite requirements.txt), then run:

    pip install streamlit
    streamlit run app.py

A browser tab opens at http://localhost:8501. The first load takes 30-60 s while it builds
the synthetic city and trains the Random Forest.
