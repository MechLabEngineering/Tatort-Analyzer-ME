Tatort-Analyzer-ME
==================

Analyze Twitter for #Tatort with Python Pandas and NLTK

![Twitter](http://mechlab-engineering.de/wordpress/wp-content/uploads/2014/05/Twitter-Analyse-300x214.png)

## Was passiert?

1. Im ersten Schritt werden mit Hilfe der StreamListener API alle Tweets zu einem Hashtag (hier: #Tatort) in eine MongoDB geschrieben
2. Im zweiten Schritt wird diese Datenbank mit Python Pandas ausgelesen und anschließend ausgewertet

## Welche Erkenntnisse?

Am besten das [IPython Notebook ansehen](http://nbviewer.ipython.org/github/MechLabEngineering/Tatort-Analyzer-ME/blob/master/AnalyzeTwitterStream.ipynb).

Mit Hilfe des Natural Language Toolkit kann wertvolle Information aus dem Stream gezogen werden.

### Handelnde Personen

Relevante Personen können identifiziert werden.

![Important People](https://raw.githubusercontent.com/MechLabEngineering/Tatort-Analyzer-ME/master/important-people-Tatort.png)

### Stimmungslage erfassen

Mit Hilfe des [SentiWS](http://asv.informatik.uni-leipzig.de/download/sentiws.html) Wortschatzes für Deutsche Sprache kann sogar ein Stimmungsbild aus den Tweets gezogen werden. Ein Naive Bayes Klassifikator, welcher mit dem SentiWS Wortschaft für positiv bzw. negativ konnotierte Wörter angelernt wurde, kann die Tweets klassifizieren.

![Stimmungsbild](https://raw.githubusercontent.com/MechLabEngineering/Tatort-Analyzer-ME/master/mood-crowd-Tatort.png)

```
R. Remus, U. Quasthoff & G. Heyer: SentiWS - a Publicly Available German-language Resource for Sentiment Analysis.
In: Proceedings of the 7th International Language Ressources and Evaluation (LREC'10), pp. 1168--1171, 2010
```
