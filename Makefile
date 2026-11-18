.PHONY: figures

figures:
	docker compose run --rm pipeline python code/06_figures.py
	docker compose run --rm pipeline python code/07_context_figures.py
	docker compose run --rm pipeline python code/08_pipeline_figure.py
