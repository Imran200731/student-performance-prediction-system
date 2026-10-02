
import csv
import json
import math
import random
from collections import Counter, defaultdict

# -----------------------------
# SETTINGS
# -----------------------------

DATA_FILE = "python_ml/student_data.csv"
MODEL_FILE = "python_ml/model.json"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "internal_score",
    "sleep_hours"
]

TARGET = "performance"
K = 5
TEST_RATIO = 0.20
RANDOM_SEED = 42


# -----------------------------
# LOAD DATASET
# -----------------------------

def load_data():
    data = []

    with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            features = [float(row[column]) for column in FEATURES]
            label = row[TARGET]

            data.append({
                "features": features,
                "label": label
            })

    return data


# -----------------------------
# STRATIFIED TRAIN/TEST SPLIT
# -----------------------------

def train_test_split(data):
    groups = defaultdict(list)

    for row in data:
        groups[row["label"]].append(row)

    random.seed(RANDOM_SEED)

    train_data = []
    test_data = []

    for label, rows in groups.items():
        random.shuffle(rows)

        test_count = max(1, round(len(rows) * TEST_RATIO))

        test_data.extend(rows[:test_count])
        train_data.extend(rows[test_count:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


# -----------------------------
# STANDARDIZATION
# -----------------------------

def calculate_scaler(train_data):
    feature_count = len(FEATURES)

    means = []
    stds = []

    for i in range(feature_count):
        values = [row["features"][i] for row in train_data]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        std = math.sqrt(variance)

        if std == 0:
            std = 1

        means.append(mean)
        stds.append(std)

    return means, stds


def scale_features(features, means, stds):
    return [
        (features[i] - means[i]) / stds[i]
        for i in range(len(features))
    ]


# -----------------------------
# KNN ALGORITHM
# -----------------------------

def euclidean_distance(a, b):
    return math.sqrt(
        sum(
            (a[i] - b[i]) ** 2
            for i in range(len(a))
        )
    )


def knn_predict(query, train_data, k):
    distances = []

    for row in train_data:
        distance = euclidean_distance(
            query,
            row["scaled_features"]
        )

        distances.append(
            (distance, row["label"])
        )

    distances.sort(key=lambda x: x[0])

    nearest = distances[:k]

    votes = Counter(label for _, label in nearest)

    prediction = votes.most_common(1)[0][0]

    return prediction


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

def confusion_matrix(actual, predicted, labels):
    matrix = {
        actual_label: {
            predicted_label: 0
            for predicted_label in labels
        }
        for actual_label in labels
    }

    for real, pred in zip(actual, predicted):
        matrix[real][pred] += 1

    return matrix


# -----------------------------
# MAIN TRAINING PROCESS
# -----------------------------

def main():

    print("=" * 55)
    print("STUDENT PERFORMANCE ML MODEL")
    print("=" * 55)

    data = load_data()

    print(f"\nDataset samples: {len(data)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Algorithm: K-Nearest Neighbors (KNN)")
    print(f"K value: {K}")

    train_data, test_data = train_test_split(data)

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Testing samples:  {len(test_data)}")

    # Calculate scaler using ONLY training data
    means, stds = calculate_scaler(train_data)

    # Scale training data
    for row in train_data:
        row["scaled_features"] = scale_features(
            row["features"],
            means,
            stds
        )

    # Scale test data
    for row in test_data:
        row["scaled_features"] = scale_features(
            row["features"],
            means,
            stds
        )

    # Predictions
    actual = []
    predicted = []

    for row in test_data:
        prediction = knn_predict(
            row["scaled_features"],
            train_data,
            K
        )

        actual.append(row["label"])
        predicted.append(prediction)

    # Accuracy
    correct = sum(
        1 for a, p in zip(actual, predicted)
        if a == p
    )

    accuracy = (correct / len(test_data)) * 100

    print(f"\nCorrect predictions: {correct}/{len(test_data)}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    labels = [
        "Poor",
        "Average",
        "Good",
        "Excellent"
    ]

    matrix = confusion_matrix(
        actual,
        predicted,
        labels
    )

    print("\nCONFUSION MATRIX")
    print("-" * 55)

    print(
        f"{'Actual / Pred':<15}"
        + "".join(f"{label:<12}" for label in labels)
    )

    for actual_label in labels:
        print(
            f"{actual_label:<15}"
            + "".join(
                f"{matrix[actual_label][predicted_label]:<12}"
                for predicted_label in labels
            )
        )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    model_training_data = []

    for row in train_data:
        model_training_data.append({
            "features": row["scaled_features"],
            "label": row["label"]
        })

    model = {
        "algorithm": "KNN",
        "k": K,
        "features": FEATURES,
        "means": means,
        "stds": stds,
        "training_data": model_training_data,
        "accuracy": round(accuracy, 2),
        "dataset_size": len(data),
        "training_size": len(train_data),
        "testing_size": len(test_data)
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    print("\nModel saved successfully!")
    print(f"File: {MODEL_FILE}")

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main() 