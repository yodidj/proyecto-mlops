	install:
		pip install --upgrade pip && pip install -r requirements.txt

	format:
		black src/

	train:	
		PYTHONPATH=. python src/train.py

	test:
		python -m unittest discover -s tests -p "*.py" || echo "Sin pruebas definidas aún."
