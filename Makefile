.PHONY : docs doctests lint tests

all : docs doctests lint tests

requirements.txt : requirements.in pyproject.toml
	pip-compile -v

lint :
	black --check .

doctests :
	sphinx-build -b doctest . docs/_build

docs :
	rm -rf docs/_build
	sphinx-build -nW . docs/_build

tests :
	pytest -v
