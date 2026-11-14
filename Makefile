# One command per stage.  `make all` regenerates everything from scratch (a few minutes on a laptop).
PY ?= python3

.PHONY: all validate experiments figures tables report docs test status clean-results

all: validate experiments tables figures report docs test status

validate:      ## correctness checks of the simulator -> results/validation/
	$(PY) -m gbsim.validate | tee results/validation.txt

experiments:   ## E1..E11 -> results/data/*.csv
	$(PY) experiments/run_all.py

tables:        ## headline numbers + markdown/csv tables -> results/tables/
	$(PY) analysis/facts.py

figures:       ## all figures -> results/figures/*.png|pdf
	$(PY) analysis/figures.py

report:        ## final PDF -> report/M1_final_report.pdf
	$(PY) report/build_report.py

docs:          ## regenerate the data-driven docs (overview, results, novelty, README block)
	$(PY) tools/build_docs.py

test:          ## regression + behaviour tests
	$(PY) tests/test_regression.py

status:        ## what has been run, headline numbers, what is left
	$(PY) tools/status.py

clean-results: ## delete generated results (keeps results/reference)
	rm -rf results/data results/figures results/tables results/validation results/validation.txt
