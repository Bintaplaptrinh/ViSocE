"""Project constants and label definitions."""

LABELS_7 = [
    "Enjoyment",
    "Sadness",
    "Anger",
    "Surprise",
    "Fear",
    "Disgust",
    "Other",
]

LABELS_6 = LABELS_7[:-1]

LABEL_MAP_7 = {label: idx for idx, label in enumerate(LABELS_7)}
LABEL_MAP_6 = {label: idx for idx, label in enumerate(LABELS_6)}
