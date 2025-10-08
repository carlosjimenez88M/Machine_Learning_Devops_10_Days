.PHONY: install run clean

install:
	conda env list | grep -q post_covid_pipeline && conda env remove -n post_covid_pipeline -y || true
	conda env create -f conda.yaml
	conda run -n post_covid_pipeline pip install --upgrade pip

run:
	conda run -n post_covid_pipeline mlflow run .
	@echo "Cleaning up temporary files..."
	@find . -type d -name "mlruns" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".hydra" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "wandb" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "outputs" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete"

clean:
	conda env remove -n post_covid_pipeline -y 2>/dev/null || true
	conda env remove -n download_data -y 2>/dev/null || true
	find . -type d -name "mlruns" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "outputs" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "wandb" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
