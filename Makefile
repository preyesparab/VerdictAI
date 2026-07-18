.PHONY: install test lint run docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	ruff check .

run:
	python app.py

docker-build:
	docker build -t repomind .

docker-up:
	docker-compose up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
