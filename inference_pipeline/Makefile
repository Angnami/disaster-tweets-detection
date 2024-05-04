.PHONY: install add  add_dev install_dev export_requirements deploy_beam run run_ui lint_check lint_fix

# ==== Install ====
install:
	@echo "Installation de l'environnement..."
	curl -sSL https://install.python-poetry.org | python3 -
	poetry env use $(shell which python3.10) && \
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install 


install_dev: install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --with dev


add:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add $(package)


add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add --group dev $(package)

# === Beam ===

export_requirements:
	@echo "Exportation des dépendances..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes

# upload_dataset_to_beam:
# 	@echo "Chargement du volume des données dans Beam..."
	
# 	beam volume upload disastertweets_dataset dataset

deploy_beam: export_requirements
	@echo "Déploiement du détecteur des tweets des catastrophes naturelles sur Beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam deploy ./tools/detector.py:run


# === RUN ===

run:
	@echo "Exécution du détecteur des tweets de catastrophes..."

	poetry run python -m tools.detector \
		--tweet "Heard about #earthquake is different cities, stay safe everyone." 

run_ui:
	cd tools/streamlit-app/ && streamlit run app.py

# === PEP8 ===
# S'assurer d'installer les dépendances dev préalablement #

lint_check:
	@echo "Vérification des problèmes de linting..."

	poetry run ruff check .

lint_fix:
	@echo "Correction des problèmes de linting..."

	poetry run ruff format .