python -m tools.data_load
python -m model.pdf_ingest
python -m model.embed_and_index
python -m model.generate_tuning_data
python -m model.tune_gemini.py
python -m app.query
echo "For app deployment see commands and setup in app/README.md"