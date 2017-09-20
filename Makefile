project := ml_playground
flake8_args := --max-line-length=130
flake8 := flake8 $(flake8_args)
pytest_args := -s --tb short --cov-config .coveragerc --cov $(project) tests
pytest := $(clay_config) py.test $(pytest_args) $(EXTRA_TEST_ARGS)

html_report := --cov-report html
test_args := --cov-report term-missing --cov-report xml --junitxml junit.xml

.DEFAULT_GOAL := bootstrap

.PHONY: bootstrap
bootstrap:
	pip install -r requirements.txt
	python setup.py develop

.PHONY: test
test: clean
	$(pytest) $(test_args)

.PHONY: clean
clean:
	@find $(project)/ "(" -name "*.pyc" -o -name "coverage.xml" -o -name "junit.xml" ")" -delete

.PHONY: lint
lint:
	$(flake8) $(project) $(project)

.PHONY: shell
shell:
	ipython

.PHONY: fmt
fmt:
	yapf ml_playground/* --recursive -e *.pyc -i