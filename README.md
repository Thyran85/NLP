# Clair — résumé et mots-clés par NLP

Clair est une application Python locale qui extrait, prétraite et analyse des documents en français. Elle prend en charge les PDF, DOCX, TXT et Markdown. Le pipeline produit un résumé, des mots-clés, des informations sur la langue et des avertissements d'extraction. L'interface existante utilise FastAPI et HTML/JavaScript plutôt que Streamlit, tout en proposant les fonctions principales du plan technique.

## Pipeline NLP

1. **Extraction** — PyMuPDF lit les PDF par blocs et ordonne les colonnes. Les pages qui contiennent moins de 50 caractères passent par Tesseract avec la langue `fra`; l'OCR peut aussi être forcée depuis l'interface. Les en-têtes et pieds de page répétés sont retirés à partir de trois pages. DOCX, TXT et MD sont aussi acceptés.
2. **Nettoyage** — le texte est normalisé en Unicode NFKC, les césures de fin de ligne sont recollées, les lignes de paragraphe réunies et les URL retirées.
3. **Prétraitement linguistique** — spaCy `fr_core_news_md` segmente en phrases et fournit lemmes et mots vides. Les phrases très courtes et majoritairement numériques sont écartées. Le texte brut est gardé pour l'affichage et les modèles génératifs; les lemmes sans mots vides alimentent les méthodes statistiques. `langdetect` signale les textes qui ne semblent pas être en français.
4. **Résumé** — sélectionnez Lead-3, TF-IDF avec bonus de position, TextRank, LSA ou LexRank (extractifs), ou BARThez / Qwen (abstractifs, optionnels). TF-IDF, TextRank, LSA et LexRank sélectionnent entre 10 et 40 % des phrases puis rétablissent leur ordre d'origine; Lead-3 prend toujours les trois premières phrases.
5. **Mots-clés** — YAKE (statistique) ou KeyBERT (similarité sémantique via `paraphrase-multilingual-MiniLM-L12-v2`). KeyBERT traite le document par blocs pour éviter de ne considérer que son début. Les doublons lemmatisés sont fusionnés, les mots vides en bordure sont supprimés et les scores renvoyés sont normalisés pour qu'un score élevé soit meilleur.
6. **Évaluation** — ROUGE-1/2/L avec un tokeniseur conservant les lettres accentuées, précision/rappel/F1@5 et @10 lemmatisés, temps par page et moyennes/écarts-types par méthode. BERTScore peut être activé à la demande.

Le détail des choix NLP et des limites est dans le [plan technique](./Plan_technique_resume_PDF_mots-cles.md). L'application garde l'architecture FastAPI du dépôt au lieu de reprendre l'arborescence Streamlit proposée par le plan. BARThez reprend le modèle abstractif recommandé par le plan; Qwen est conservé comme option déjà présente. La méthode par défaut est TextRank.

## Installation

Python 3.10 ou plus récent est recommandé. PyTorch s'installe séparément pour choisir la distribution CPU; les autres dépendances applicatives et NLP sont dans `requirements.txt`.

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download fr_core_news_md
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download fr_core_news_md
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Ouvrez <http://127.0.0.1:8000>. Si `.venv` existe déjà, activez-le sans le recréer.

### OCR sur PDF scannés

Tesseract est un programme système séparé de Python; installez-le ainsi que les données de langue française :

- Debian/Ubuntu : `sudo apt install tesseract-ocr tesseract-ocr-fra`
- macOS avec Homebrew : `brew install tesseract tesseract-lang`
- Windows : installez Tesseract OCR et vérifiez que `tesseract.exe` est dans le `PATH`; ajoutez les données `fra`.

L'interface affiche les pages traitées par OCR dans les avertissements. Sans Tesseract, un PDF scanné reçoit une erreur explicite plutôt qu'un faux résultat vide.

## Utilisation

Choisissez le fichier, la méthode de résumé, le taux extractif, la méthode de mots-clés et le nombre de candidats. Activez « Forcer l'OCR » uniquement pour un PDF. Le résultat comprend le résumé, les mots-clés avec leur score et surlignés dans le texte, le texte nettoyé, la langue détectée, les avertissements, le nombre de pages et le temps de traitement. Les exports disponibles sont TXT, JSON, Word et impression PDF depuis le navigateur.

Les principales routes sont :

- `GET /api/health` (ou `GET /health`) : disponibilité des dépendances NLP, du modèle spaCy, de KeyBERT et de Tesseract.
- `POST /api/summarize` : fichier multipart et paramètres `method`, `rate`, `keyword_method`, `top_k`, `length` et `force_ocr`.
- `POST /api/export/docx` : création d'un document Word.
- `POST /api/evaluate` : évaluation d'une liste de résultats annotés; ajouter `?bertscore=true` pour calculer aussi BERTScore.

Un premier usage de BARThez, Qwen ou KeyBERT télécharge les poids depuis Hugging Face dans le cache local; l'inférence et l'analyse du document restent locales. Les documents ne sont pas envoyés à une API de résumé distante. Les modèles abstractifs peuvent être lents sur CPU et les longs textes nécessitent plusieurs passes. Les fichiers temporaires d'extraction sont supprimés après lecture.

## Configuration

Copiez `.env.example` vers `.env` à la racine pour modifier le modèle local, la limite de téléversement, la taille des blocs Qwen ou le cache Hugging Face :

```dotenv
LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
MAX_UPLOAD_MB=250
CHUNK_CHARS=36000
HF_HOME=.model-cache
```

Les PDF protégés par mot de passe ou sans texte exploitable sont refusés. L'OCR est réservée aux pages scannées ou explicitement forcées. La sortie de tout résumé génératif doit être relue.

## Évaluation sur un corpus annoté

Préparez un JSON contenant une liste d'objets avec un résumé produit et sa référence, par exemple :

```json
[
  {
    "fichier": "rapport.pdf",
    "methode_resume": "textrank",
    "methode_mots_cles": "yake",
    "resume": "Résumé produit...",
    "resume_reference": "Résumé annoté...",
    "mots_cles": [{"terme": "gestion des risques", "score": 0.82}],
    "mots_cles_reference": ["gestion du risque"],
    "duree_s": 3.4,
    "nb_pages": 12
  }
]
```

Lancez l'évaluation et écrivez les mesures détaillées, moyennes et écarts-types dans un CSV :

```bash
python -m app.evaluation corpus.json resultats.csv
```

Pour faire exécuter chaque méthode extractive et chaque méthode de mots-clés sur chaque document, le JSON doit contenir le texte dans `texte` (ou `text`), `resume_reference` et `mots_cles_reference` :

```bash
python -m app.evaluation corpus_a_benchmarker.json comparaison.csv --benchmark
```

Le benchmark ajoute BARThez et Qwen avec `--include-abstractive`; ces modèles téléchargent leurs poids au premier usage et ralentissent fortement l'évaluation CPU.

Pour ajouter BERTScore (téléchargement d'un modèle sémantique requis au premier lancement) :

```bash
python -m app.evaluation corpus.json resultats.csv --bertscore
```

Les dépendances de test s'installent avec `pip install -r requirements-dev.txt`, puis les tests se lancent avec `python -m pytest`.
