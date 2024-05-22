default:
	pip install .

install-dev:
	pip install -e ".[dev]"

.PHONY: tests
tests:
	pytest -v --cov=process_blockwise process_blockwise
	flake8 process_blockwise
