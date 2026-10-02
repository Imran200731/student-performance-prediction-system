import json
import math
from collections import Counter

MODEL_FILE = "python_ml/model.json"


def load_model():
    with open(MODEL_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def scale_features(features, means, stds):
    return [
        (features[i] - means[i]) / stds[i]
        for i in range(len(features))
    ]


def euclidean_distance(a, b):
    return math.sqrt(
        sum(
            (a[i] - b[i]) ** 2
            for i in range(len(a))
        )
    )


def predict(features, model):

    scaled_input = scale_features(
        features,
        model["means"],
        model["stds"]
    )

    distances = []

    for row in model["training_data"]:

        distance = euclidean_distance(
            scaled_input,
            row["features"]
        )

        distances.append(
            (distance, row["label"])
        )

    distances.sort(key=lambda x: x[0])

    nearest = distances[:model["k"]]

    votes = Counter(
        label for _, label in nearest
    )

    prediction, vote_count = votes.most_common(1)[0]

    confidence = (vote_count / model["k"]) * 100

    return prediction, confidence


# Load trained model
model = load_model()

# Example student
student = [
    8,      # study hours
    95,     # attendance
    92,     # previous score
    94,     # assignment score
    91,     # internal score
    8       # sleep hours
]

prediction, confidence = predict(
    student,
    model
)

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 50)

print("\nStudent input:")
print("Study Hours:", student[0])
print("Attendance:", student[1], "%")
print("Previous Score:", student[2])
print("Assignment Score:", student[3])
print("Internal Score:", student[4])
print("Sleep Hours:", student[5])

print("\nPrediction:", prediction)
print("Confidence:", f"{confidence:.2f}%")

print("=" * 50)