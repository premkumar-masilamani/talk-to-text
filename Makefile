.PHONY: setup
setup:
	@echo "Creating virtual environment..."
	pipenv install

.PHONY: run
run:
	pipenv run python -m transcriber.main -v
