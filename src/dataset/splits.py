from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

def create_standard_split(samples):
    train_val, test = train_test_split(
        samples,
        test_size=0.1,
        random_state=42,
        stratify=[y for _, y in samples]
    )

    train, val = train_test_split(
        train_val,
        test_size=0.111,
        random_state=42,
        stratify=[y for _, y in train_val]
    )

    return train, val, test

def create_generalization_split(samples):
    class_map = defaultdict(list)

    for img, label in samples:
        class_map[label].append((img, label))

    classes = list(class_map.keys())
    random.seed(42)
    random.shuffle(classes)

    split = int(0.9 * len(classes))

    train_classes = set(classes[:split])
    test_classes = set(classes[split:])

    train, test = [], []

    for c in train_classes:
        train.extend(class_map[c])

    for c in test_classes:
        test.extend(class_map[c])

    return train, test