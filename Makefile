.PHONY : build docs doctests lint tests

all : build docs doctests lint tests

requirements.txt : requirements.in pyproject.toml
	pip-compile -v

lint :
	black --check .

doctests :
	sphinx-build -b doctest . docs/_build

docs :
	rm -rf docs/_build
	sphinx-build -nW --keep-going . docs/_build

tests :
	pytest -v

build :
	python -m build
	twine check dist/*.tar.gz dist/*.whl
