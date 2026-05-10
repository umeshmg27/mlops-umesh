.PHONY: setup data eda train test lint api ui docker-build compose-up k8s-apply clean-artifacts

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

data:
	python scripts/download_data.py

eda:
	python scripts/run_eda.py

train:
	python scripts/train_model.py

test:
	pytest

lint:
	ruff check .

api:
	uvicorn heart_disease_mlops.api:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501

docker-build:
	docker build -t heart-disease-mlops:latest .

compose-up:
	docker compose up --build

k8s-apply:
	kubectl apply -f k8s/

clean-artifacts:
	rm -rf mlruns reports/figures/*.png reports/screenshots/*.png models/*.joblib models/*.json data/raw/*.csv data/processed/*.csv

