# Detekcja obiektów na obrazach — YOLO, Faster R-CNN oraz RF-DETR

Projekt porównujący trzy rodziny architektur detekcji obiektów na zbiorze danych z kategorii mody.  
Realizowany w ramach kursu **Sztuczne Sieci Neuronowe**, AGH WFiIS.

**Autorzy:** Yuliya Zviarko · Bartłomiej Obrochta · Sebastian Zarębski

---

## Zbiór danych

[Colorful Fashion Dataset for Object Detection](https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection/data) — ~2680 obrazów, 10 klas:

`sunglass` · `hat` · `jacket` · `shirt` · `pants` · `shorts` · `skirt` · `dress` · `bag` · `shoe`

Adnotacje w formacie PASCAL VOC.

---

## Struktura repozytorium

```
├── YOLO/               # Notebooki i eksperymenty z rodziną YOLO
├── fastrcnn/           # Implementacja i trening Faster R-CNN
├── RFDetr/             # Implementacja i trening RF-DETR
├── dataset/            # Skrypty do pobrania i podziału zbioru danych
├── visualizations/     # Krzywe uczenia i wykresy porównawcze
├── prezentacja/        # Slajdy prezentacji (LaTeX/Beamer)
├── sprawozdanie/       # Dokumentacja projektu (LaTeX)
├── requirements.txt    # Zależności Python
└── pyproject.toml
```
