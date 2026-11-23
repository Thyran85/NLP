# Guide du projet NLP : réalisation, traitement et écarts avec le plan

## 1. À quoi sert ce document ?

Ce guide complète le [README](./README.md). Le README permet surtout d'installer et de lancer l'application. Ici, on explique :

- ce que proposait le [plan technique initial](./Plan_technique_resume_PDF_mots-cles.md) ;
- ce qui a réellement été développé et les choix adaptés au dépôt existant ;
- comment un document traverse le pipeline NLP ;
- quels modèles et méthodes sont utilisés ;
- comment lancer, tester et évaluer le projet, et ce qu'il reste à valider.

Le plan technique est une spécification de départ, pas une description exacte de tous les fichiers actuels. Il reste inchangé.

## 2. Plan initial et projet réalisé

| Sujet | Proposition du plan | Réalisation actuelle |
|---|---|---|
| Interface | Créer une application Streamlit (`app.py`). | Conserver l'application FastAPI du dépôt et son interface HTML/JavaScript dans `static/index.html`. L'application se lance avec Uvicorn. |
| Organisation | Modules nommés `extraction.py`, `pretraitement.py`, `resume.py`, `motscles.py`, `evaluation.py`. | Fonctions réparties dans `app/summarizer.py`, `app/preprocessing.py`, `app/nlp.py` et `app/evaluation.py`, avec orchestration dans `app/pipeline.py` et API dans `app/main.py`. |
| Documents | Principalement les PDF. | PDF, DOCX, TXT et Markdown. Les PDF sont lus par blocs, les colonnes sont ordonnées et les en-têtes/pieds de page répétés sont filtrés. |
| OCR | OCR Tesseract en français pour les pages scannées, avec possibilité de la forcer. | Implémentée pour les pages PDF de moins de 50 caractères ou lorsque l'option est activée. Tesseract et les données de langue française doivent être installés séparément. |
| Prétraitement | Normalisation Unicode, correction des mots coupés, segmentation, filtres, lemmatisation et signalement de langue. | Ces étapes sont implémentées avec spaCy `fr_core_news_md`; `langdetect` signale une langue différente du français, mais le document n'est pas rejeté automatiquement. |
| Résumé extractif | Lead-3, TF-IDF + position, TextRank, LSA et LexRank. | Les cinq méthodes sont disponibles. LSA et LexRank sont calculés dans le projet avec scikit-learn et NetworkX, plutôt qu'avec la bibliothèque `sumy` proposée comme option dans le plan. |
| Résumé abstractif | BARThez en option, appliqué par blocs. | BARThez est disponible, avec découpage et synthèses successives; Qwen, déjà présent dans le projet, est conservé comme option supplémentaire. Ces modèles demandent PyTorch et téléchargent leurs poids au premier usage. |
| Mots-clés | YAKE et KeyBERT, dédoublonnage par lemmes et scores normalisés. | YAKE et KeyBERT sont disponibles. KeyBERT traite des blocs de texte puis fusionne les candidats; le plan suggérait plutôt de moyenner les représentations des blocs. |
| Évaluation | ROUGE, BERTScore, P/R/F1 des mots-clés, durée par page et comparaison sur des corpus annotés. | Les métriques et les commandes de benchmark sont implémentées. Aucun corpus annoté OrangeSum, MLSUM, DEFT ou corpus manuel n'est fourni avec le projet : il faut fournir ses propres références pour obtenir une évaluation expérimentale. |
| Tests et déploiement | Tests par module, et Docker éventuellement. | Des tests pytest couvrent les fonctions principales et l'API. Le lancement documenté est local avec Uvicorn; Docker et l'interface Streamlit du plan ne font pas partie de cette réalisation. |

### Ce que signifie « suivre le plan »

Les principales étapes NLP et les méthodes visées par le plan sont présentes. Les différences d'architecture (FastAPI au lieu de Streamlit) et de découpage des fichiers permettent de s'intégrer au projet déjà existant, sans réécrire son interface. Quelques recommandations du plan ne sont pas strictement identiques dans leur implémentation — notamment le traitement de KeyBERT par blocs — et les corpus d'évaluation restent à fournir.

## 3. Organisation du code

| Fichier | Responsabilité |
|---|---|
| [`app/main.py`](./app/main.py) | Routes FastAPI, validation des paramètres, réception des fichiers, statut de l'application et exports. |
| [`app/summarizer.py`](./app/summarizer.py) | Extraction PDF/DOCX/TXT/Markdown, OCR PDF optionnelle et découpage de texte. |
| [`app/preprocessing.py`](./app/preprocessing.py) | Nettoyage, détection de langue, segmentation en phrases et normalisation par lemmatisation spaCy. |
| [`app/nlp.py`](./app/nlp.py) | Méthodes de résumé et extraction de mots-clés. |
| [`app/local_model.py`](./app/local_model.py) | Chargement local et génération avec BARThez ou Qwen. |
| [`app/pipeline.py`](./app/pipeline.py) | Enchaînement prétraitement → résumé → mots-clés, puis préparation du résultat. |
| [`app/evaluation.py`](./app/evaluation.py) | Métriques, évaluation d'un corpus et benchmark en ligne de commande. |
| [`app/config.py`](./app/config.py) | Configuration `.env` et emplacement du cache des modèles. |
| [`static/index.html`](./static/index.html) | Interface de dépôt de document, réglage des méthodes et affichage des résultats. |
| [`tests/test_nlp.py`](./tests/test_nlp.py) | Tests d'extraction, prétraitement, méthodes NLP, API et évaluation. |

### Techniques d'application dans l'interface et l'API

L'interface envoie le fichier et les options choisies à `POST /api/summarize` sous forme de formulaire multipart. FastAPI vérifie le format, la taille et les valeurs des paramètres, puis écrit le fichier dans un fichier temporaire pour le traiter. Le fichier temporaire est supprimé après l'extraction. Les traitements bloquants sont exécutés dans un thread afin de ne pas bloquer la boucle asynchrone du serveur.

Les erreurs sont renvoyées avec un statut adapté : format incorrect (415), paramètres ou document invalides (422), fichier trop volumineux (413), dépendance ou modèle indisponible (503). `GET /api/health` permet de vérifier les dépendances principales, le modèle spaCy, KeyBERT et Tesseract. Les modèles lourds sont chargés à la demande et gardés en mémoire pour éviter de les recharger à chaque requête.

## 4. Comment un document est traité

Le flux d'une analyse est le suivant :

```text
PDF / DOCX / TXT / MD
        ↓
Extraction du texte (OCR si nécessaire pour un PDF)
        ↓
Nettoyage et prétraitement français
        ↓
Résumé choisi + extraction des mots-clés choisie
        ↓
Résultat JSON → interface, API ou export
```

### Étape 1 — Extraction

- **PDF** : PyMuPDF lit le texte par blocs et ordonne les colonnes. Les lignes qui semblent être des en-têtes ou pieds de page répétés sont éliminées.
- **PDF scanné** : une page contenant moins de 50 caractères passe par Tesseract en français. L'OCR peut aussi être forcée depuis l'interface.
- **DOCX** : les paragraphes et les tableaux sont lus.
- **TXT/Markdown** : le texte UTF-8 est lu directement.

Un PDF protégé ou un document sans texte exploitable produit une erreur explicite plutôt qu'un résultat de résumé vide.

**Algorithme PDF, plus précisément :**

1. Lire les blocs de texte avec PyMuPDF et ignorer les blocs qui ne sont pas du texte.
2. Trier les blocs en plaçant d'abord la colonne gauche, puis la droite, et en ordonnant chaque colonne verticalement.
3. Si l'OCR est activée ou si la page contient moins de 50 caractères, rendre la page en image à 300 dpi et appeler Tesseract avec la langue `fra`.
4. À partir de trois pages, considérer les deux premières et les deux dernières lignes comme marges possibles. Ignorer les chiffres lors de leur comparaison et supprimer les lignes retrouvées sur plus de la moitié des pages.

Ce traitement de mise en page est une heuristique : les PDF atypiques, tableaux complexes et colonnes irrégulières doivent être vérifiés dans la sortie du texte extrait.

### Étape 2 — Prétraitement linguistique

Le texte est normalisé en Unicode, les césures de fin de ligne sont recollées, les lignes d'un paragraphe sont réunies et les URL sont retirées. spaCy découpe ensuite le texte en phrases. Les phrases très courtes et celles principalement constituées de nombres sont filtrées.

Deux représentations servent à des buts différents :

- les phrases originales restent lisibles et peuvent être sélectionnées telles quelles par les méthodes extractives ;
- une représentation lemmatisée et débarrassée des mots vides sert aux calculs statistiques et aux similarités.

La détection de langue avertit l'utilisateur si le document semble ne pas être en français. Le traitement continue néanmoins avec les ressources françaises; la qualité peut donc être inférieure sur une autre langue.

**Détails du prétraitement :**

- `unicodedata.normalize("NFKC", ...)` unifie les formes Unicode, notamment certaines ligatures.
- Des expressions régulières recollent les mots coupés par une césure et réunissent les lignes d'un même paragraphe; les URL sont retirées.
- `langdetect` intervient seulement à partir de 40 caractères et sa graine est fixée afin de rendre la détection reproductible.
- spaCy charge `fr_core_news_md` et désactive la reconnaissance d'entités nommées, qui n'est pas nécessaire au pipeline. Le découpage utilise les phrases `doc.sents`.
- Une phrase est écartée si elle contient moins de cinq mots hors espaces et ponctuation, si au moins 60 % des mots sont numériques, ou si elle ne fournit aucun lemme alphabétique exploitable.
- Pour la représentation normalisée, les lemmes sont mis en minuscules et les mots vides et ponctuations sont ignorés.

Le recollage automatique ne peut pas toujours distinguer une césure typographique d'un mot composé; les mots concernés doivent être contrôlés sur les documents importants.

### Étape 3 — Résumé

| Méthode | Type | Principe |
|---|---|---|
| `lead3` | Extractif | Reprend les trois premières phrases, sans dépendre du taux choisi. |
| `tfidf` | Extractif | Classe les phrases selon leurs termes importants, avec un bonus de position. |
| `textrank` | Extractif | Utilise PageRank sur un graphe de similarité entre phrases. |
| `lexrank` | Extractif | Classe les phrases par leur similarité avec les autres phrases. |
| `lsa` | Extractif | Utilise une réduction de dimension pour estimer l'importance des phrases. |
| `barthez` | Abstractif | Modèle français de génération; traite les textes en blocs et condense les synthèses intermédiaires. |
| `qwen` | Abstractif | Modèle génératif local conservé comme option; les longues entrées sont traitées par blocs. |

Pour les méthodes extractives sauf Lead-3, le taux réglable est de 10 à 40 %. Les phrases retenues sont remises dans leur ordre d'origine.

**Calcul des résumés extractifs :**

- Le nombre de phrases à retenir est l'arrondi supérieur de `nombre_de_phrases × taux`, borné entre une phrase et le nombre total de phrases.
- **TF-IDF + position** : calculer une matrice TF-IDF sur les phrases normalisées, sommer les poids de chaque phrase puis ajouter un bonus qui décroît de la première à la dernière phrase. Le bonus maximal vaut 10 % du score TF-IDF maximal.
- **TextRank** : calculer la similarité cosinus entre les vecteurs TF-IDF de chaque paire de phrases, retirer les liens d'une phrase vers elle-même, puis utiliser PageRank pondéré sur ce graphe.
- **LexRank** : utiliser la même matrice de similarité, retirer les liens inférieurs à 0,1 et classer les phrases par somme des similarités conservées. Si ce seuil élimine tous les liens, refaire le classement sans seuil.
- **LSA** : réduire la matrice TF-IDF avec `TruncatedSVD` (au plus cinq composantes), puis classer selon la norme du vecteur réduit. Si la matrice est trop petite pour une réduction, utiliser la somme TF-IDF.
- En cas d'égalité, l'ordre initial des phrases est conservé autant que possible; après le classement, le résumé final est toujours restitué dans l'ordre du document.

Ces méthodes sélectionnent des phrases existantes; elles ne reformulent pas le texte et peuvent produire des raccords abrupts entre phrases.

### Étape 4 — Mots-clés

- **YAKE** repère statistiquement les expressions candidates; son score initial est inversé et normalisé afin qu'un score de sortie plus élevé signifie un meilleur classement.
- **KeyBERT** utilise le modèle d'embeddings multilingue `paraphrase-multilingual-MiniLM-L12-v2` pour classer les expressions par similarité sémantique. Les textes sont traités par blocs; les candidats sont ensuite dédoublonnés et normalisés.

Les candidats sont comparés après normalisation des lemmes afin de limiter les variantes redondantes. Le nombre demandé est réglable dans l'interface.

**Paramètres et fusion des candidats :**

- YAKE travaille en français, recherche des groupes de 1 à 3 mots, demande jusqu'à trois fois le nombre de candidats voulu et utilise `dedupLim=0.9`. Son score natif est transformé par `1 / (1 + score)` pour que les meilleurs résultats aient un score élevé.
- KeyBERT découpe le texte en blocs d'environ 800 caractères. Dans chaque bloc, il cherche des groupes de 1 à 3 mots, utilise les mots vides français issus de spaCy, active MMR avec une diversité de 0,5 et retient jusqu'à `top_k` candidats.
- Les candidats des blocs sont réunis, lemmatisés et dédoublonnés. Pour une même forme lemmatisée, le meilleur score est gardé; le classement final est tronqué à `top_k`.

Le traitement de KeyBERT par blocs permet d'examiner les différentes parties d'un document long. Il ne correspond pas exactement à l'approche du plan, qui proposait de moyenner les embeddings des blocs.

### Étape 5 — Résumé abstractif

BARThez et Qwen génèrent du texte au lieu de choisir uniquement des phrases existantes. Ils sont chargés à la demande et exécutés localement sur CPU dans la configuration par défaut.

- **BARThez** : les phrases sont regroupées dans des blocs d'environ 2 700 caractères. Chaque bloc est d'abord résumé, puis les synthèses sont fusionnées et condensées si elles dépassent la limite, avant de générer le résumé final. Le tokenizer du modèle limite chaque entrée à 1 024 tokens.
- **Qwen** : le texte est découpé selon `CHUNK_CHARS` (36 000 caractères par défaut). Le tokenizer applique également une limite d'entrée de 12 000 tokens; la génération est déterministe (`do_sample=False`).
- Pour les deux modèles, le paramètre de longueur propose les profils court, moyen et détaillé. Les textes très longs peuvent nécessiter plusieurs passes et l'inférence CPU peut être lente.

Un résumé génératif peut reformuler ou omettre des détails : il doit être relu et comparé au document source.

### Étape 6 — Résultat

Le résultat fournit notamment le résumé, les mots-clés et leurs scores, la langue détectée, les avertissements, les informations de pages et de durée, ainsi que le texte extrait et prétraité. L'interface permet de consulter et d'exporter les résultats.

## 5. Modèles et dépendances à distinguer

| Ressource | Utilisation | Installation / téléchargement |
|---|---|---|
| spaCy `fr_core_news_md` | Segmentation française, lemmatisation et mots vides. | Installé séparément avec `python -m spacy download fr_core_news_md`. |
| `langdetect` | Détection indicative de la langue. | Dépendance Python du projet. |
| YAKE | Extraction statistique de mots-clés. | Dépendance Python du projet; pas de poids neuronaux à télécharger. |
| KeyBERT + Sentence Transformers | Classement sémantique de mots-clés. | Les poids de `paraphrase-multilingual-MiniLM-L12-v2` sont téléchargés au premier usage. |
| BARThez | Résumé abstractif en français. | PyTorch et Transformers; poids Hugging Face téléchargés au premier usage. |
| Qwen | Résumé abstractif local, optionnel. | PyTorch et Transformers; modèle configurable avec `LOCAL_MODEL`, poids téléchargés au premier usage. |
| Tesseract | OCR des pages PDF scannées. | Programme système séparé avec les données de langue `fra`; il est facultatif pour les autres formats et PDF textuels. |
| BERTScore | Mesure facultative de similarité sémantique en évaluation. | Dépendance et modèle de mesure supplémentaires, requis seulement avec `--bertscore`. |

Les modèles génératifs et KeyBERT peuvent être lents sur CPU. Les documents et les inférences restent traités localement; les téléchargements de poids nécessitent une connexion lors du premier usage.

## 6. Installation et démarrage

### Linux / macOS

À faire une fois, depuis la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

À chaque démarrage :

```bash
cd /home/thyran/Project/proto
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Ouvrir ensuite <http://127.0.0.1:8000>. Si le dossier `.venv` n'existe pas, il faut d'abord exécuter les commandes d'installation ci-dessus. Arrêter le serveur avec `Ctrl+C`.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download fr_core_news_md
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### OCR sous Linux

Pour analyser des PDF scannés, installer Tesseract et ses données françaises en plus des dépendances Python :

```bash
sudo apt install tesseract-ocr tesseract-ocr-fra
```

L'API de statut est accessible sur `GET /api/health` et `GET /health`; ces deux routes renvoient le même état.

## 7. Utiliser et évaluer le pipeline

### Depuis l'interface

1. Ouvrir <http://127.0.0.1:8000>.
2. Déposer un PDF, DOCX, TXT ou MD.
3. Choisir la méthode de résumé, le taux extractif, la méthode de mots-clés et leur nombre.
4. Activer l'OCR uniquement pour un PDF si nécessaire.
5. Cliquer sur **Analyser le document**, puis examiner le résumé, les mots-clés et les avertissements.

### Depuis l'API

- `POST /api/summarize` accepte un fichier multipart et les paramètres `method`, `rate`, `keyword_method`, `top_k`, `length` et `force_ocr`.
- `POST /api/export/docx` produit un document Word.
- `POST /api/evaluate` calcule les métriques sur des résultats et références fournis.
- `GET /api/health` et `GET /health` indiquent la disponibilité des dépendances et ressources principales.

### Évaluation avec références annotées

Pour évaluer des résultats déjà produits, chaque entrée JSON doit contenir au minimum un résumé produit (`resume`), un résumé de référence (`resume_reference`), et peut inclure les mots-clés et leur référence :

```json
[
  {
    "fichier": "rapport.pdf",
    "methode_resume": "textrank",
    "methode_mots_cles": "yake",
    "resume": "Résumé produit...",
    "resume_reference": "Résumé de référence...",
    "mots_cles": [{"terme": "analyse linguistique", "score": 0.8}],
    "mots_cles_reference": ["analyse linguistique"],
    "duree_s": 2.5,
    "nb_pages": 4
  }
]
```

Enregistrer le tableau de bord au format CSV :

```bash
python -m app.evaluation corpus.json resultats.csv
```

#### Mesures calculées et interprétation

- **ROUGE-1** compare les mots communs; **ROUGE-2**, les suites de deux mots; **ROUGE-L**, la plus longue sous-séquence commune. Pour chaque mesure, le programme calcule précision, rappel et F1. Le tokenizer français utilise `\w+` et conserve les lettres accentuées, afin d'éviter de fausser les scores sur des mots comme « été ».
- **BERTScore**, activé avec `--bertscore`, mesure la similarité sémantique entre les tokens du résumé candidat et ceux de la référence avec un modèle multilingue/français. Il complète ROUGE mais ne remplace pas l'évaluation humaine.
- **P@k, R@k et F1@k** comparent les mots-clés produits avec les mots-clés de référence après lemmatisation et dédoublonnage. La comparaison est faite pour `k=5` et `k=10`. La précision est le nombre de correspondances divisé par les mots-clés prédits; le rappel est le nombre de correspondances divisé par les mots-clés de référence; le F1 est leur moyenne harmonique.
- **Secondes par page** est calculé si la durée et le nombre de pages sont présents. Pour un benchmark, la durée enregistrée inclut le prétraitement, le résumé et l'extraction des mots-clés.
- Le CSV contient les résultats individuels puis, pour chaque couple de méthodes, la moyenne et l'écart-type des mesures. L'écart-type utilise la formule échantillonnale et vaut zéro s'il n'y a qu'un seul résultat.

Un score élevé n'indique pas à lui seul qu'un résumé est fidèle, clair ou utile. Il faut employer les mêmes documents et références pour comparer les méthodes, vérifier plusieurs exemples manuellement et signaler la taille du corpus dans le rapport. Des annotations de référence de qualité et cohérentes entre annotateurs sont essentielles.

Pour lancer toutes les méthodes sur le texte source, le corpus doit aussi fournir `texte` (ou `text`), `resume_reference` et `mots_cles_reference` :

```bash
python -m app.evaluation corpus_a_benchmarker.json comparaison.csv --benchmark
```

Sans option supplémentaire, le benchmark exécute cinq méthodes extractives avec YAKE et KeyBERT, soit dix combinaisons par document. Ajouter `--include-abstractive` ajoute BARThez et Qwen, soit quatorze combinaisons; `--bertscore` ajoute la mesure BERTScore. Ces options peuvent télécharger des modèles et être lentes. Aucun jeu de données annoté n'est livré avec le projet.

## 8. État d'avancement et suites possibles

### Réalisé

- Pipeline extraction → prétraitement → résumé et mots-clés → réponse API/interface.
- Méthodes extractives et optionnelles abstractives, avec paramètres de sélection.
- Tests couvrant les cas principaux; la suite a été vérifiée avec 18 tests réussis.
- Commandes d'évaluation et métriques intégrées.

### À valider ou compléter selon les objectifs du rapport

1. Installer Tesseract et tester sur un véritable PDF scanné; les tests automatisés du chemin OCR utilisent un OCR simulé.
2. Télécharger et évaluer réellement les poids BARThez dans l'environnement cible.
3. Constituer un corpus français avec résumés et mots-clés de référence; le plan cite OrangeSum/MLSUM, DEFT 2016 et un corpus annoté manuellement comme pistes.
4. Exécuter le benchmark et commenter les résultats plutôt que de conclure à partir des seuls tests logiciels.
5. Mesurer les temps sur le matériel cible et décider si les méthodes abstractives sont adaptées à l'usage prévu.
6. Vérifier les mots coupés et mots composés après extraction; leur recollage automatique peut parfois les modifier.

Les tests prouvent le comportement des composants testés; ils ne remplacent ni une validation OCR sur des scans réels, ni une évaluation de qualité sur un corpus annoté.
