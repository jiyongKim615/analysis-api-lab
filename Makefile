PYTHON ?= python3

.PHONY: lab test evaluate demo serve
lab: test
	$(PYTHON) -m compileall -q analysis_api scripts tests
	$(PYTHON) -m scripts.evaluate

test:
	$(PYTHON) -m unittest discover -s tests -v

evaluate:
	$(PYTHON) -m scripts.evaluate

demo:
	$(PYTHON) -m analysis_api demo --out output/demo.json

serve:
	CX_PROVIDER=offline CX_ALLOW_EXTERNAL=false $(PYTHON) -m uvicorn analysis_api.api:app --host 127.0.0.1 --port 8000
