.PHONY: install install-dev scrape analyze visualize train all test clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

scrape:
	python -m lcw_analytics scrape

analyze:
	python -m lcw_analytics analyze

visualize:
	python -m lcw_analytics visualize

train:
	python scripts/train_sentiment.py --model tfidf

all: scrape analyze visualize

test:
	pytest tests/ -v

clean:
	rm -rf outputs/figures/*.png outputs/reports/*.html
	rm -rf __pycache__ src/**/__pycache__ tests/**/__pycache__
	find . -name "*.pyc" -delete