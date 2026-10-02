let model = null;

// ===============================
// LOAD PYTHON TRAINED MODEL
// ===============================

async function loadModel() {
    try {
        const response = await fetch("python_ml/model.json");

        if (!response.ok) {
            throw new Error("Could not load model.json");
        }

        model = await response.json();

        console.log("ML model loaded successfully!");
        console.log("Algorithm:", model.algorithm);
        console.log("K value:", model.k);
        console.log("Training samples:", model.training_size);
        console.log("Model accuracy:", model.accuracy + "%");

    } catch (error) {
        console.error("Model loading error:", error);
        alert("ML model could not be loaded. Please run the website using Live Server.");
    }
}


// ===============================
// STANDARDIZATION
// ===============================

function scaleFeatures(features) {
    return features.map((value, index) => {
        return (
            (value - model.means[index]) /
            model.stds[index]
        );
    });
}


// ===============================
// EUCLIDEAN DISTANCE
// ===============================

function euclideanDistance(a, b) {

    let sum = 0;

    for (let i = 0; i < a.length; i++) {
        sum += Math.pow(a[i] - b[i], 2);
    }

    return Math.sqrt(sum);
}


// ===============================
// KNN PREDICTION
// ===============================

function predictWithKNN(inputFeatures) {

    const scaledInput = scaleFeatures(inputFeatures);

    const distances = [];

    for (const row of model.training_data) {

        const distance = euclideanDistance(
            scaledInput,
            row.features
        );

        distances.push({
            distance: distance,
            label: row.label
        });
    }

    // Sort by nearest distance
    distances.sort(
        (a, b) => a.distance - b.distance
    );

    // Select K nearest neighbors
    const nearest = distances.slice(0, model.k);

    // Count votes
    const votes = {};

    for (const item of nearest) {

        if (!votes[item.label]) {
            votes[item.label] = 0;
        }

        votes[item.label]++;
    }

    // Find majority class
    let prediction = "";
    let highestVotes = 0;

    for (const label in votes) {

        if (votes[label] > highestVotes) {
            highestVotes = votes[label];
            prediction = label;
        }
    }

    // Confidence
    const confidence =
        (highestVotes / model.k) * 100;

    return {
        prediction: prediction,
        confidence: confidence
    };
}


// ===============================
// MAIN PREDICTION FUNCTION
// ===============================

async function predictPerformance() {

    // Load model if not already loaded
    if (!model) {
        await loadModel();
    }

    if (!model) {
        return;
    }

    // Get input values
    const studyHours =
        parseFloat(document.getElementById("studyHours").value);

    const attendance =
        parseFloat(document.getElementById("attendance").value);

    const previousScore =
        parseFloat(document.getElementById("previousScore").value);

    const assignmentScore =
        parseFloat(document.getElementById("assignmentScore").value);

    const internalScore =
        parseFloat(document.getElementById("internalScore").value);

    const sleepHours =
        parseFloat(document.getElementById("sleepHours").value);


    // ===============================
    // VALIDATION
    // ===============================

    if (
        isNaN(studyHours) ||
        isNaN(attendance) ||
        isNaN(previousScore) ||
        isNaN(assignmentScore) ||
        isNaN(internalScore) ||
        isNaN(sleepHours)
    ) {

        alert("Please fill in all fields.");
        return;
    }


    if (
        studyHours < 0 ||
        studyHours > 24 ||
        attendance < 0 ||
        attendance > 100 ||
        previousScore < 0 ||
        previousScore > 100 ||
        assignmentScore < 0 ||
        assignmentScore > 100 ||
        internalScore < 0 ||
        internalScore > 100 ||
        sleepHours < 0 ||
        sleepHours > 24
    ) {

        alert("Please enter valid values.");
        return;
    }


    // ===============================
    // CREATE FEATURE VECTOR
    // ===============================

    const inputFeatures = [
        studyHours,
        attendance,
        previousScore,
        assignmentScore,
        internalScore,
        sleepHours
    ];


    // ===============================
    // PREDICT
    // ===============================

    const result =
        predictWithKNN(inputFeatures);

    const prediction =
        result.prediction;

    const confidence =
        result.confidence;


    // ===============================
    // DISPLAY RESULT
    // ===============================

    document.getElementById("result").style.display = "block";

    document.getElementById("prediction").textContent =
        prediction;

    document.getElementById("score").textContent =
        confidence.toFixed(0) + "%";

    document.getElementById("scoreFill").style.width =
        confidence + "%";


    // ===============================
    // RESULT MESSAGE
    // ===============================

    const message =
        document.getElementById("message");

    if (prediction === "Excellent") {

        message.textContent =
            "Outstanding academic indicators! Keep maintaining these strong study habits and consistency. 🚀";

    }

    else if (prediction === "Good") {

        message.textContent =
            "Good academic performance indicators. A little more consistency could push the student even higher. 📈";

    }

    else if (prediction === "Average") {

        message.textContent =
            "The indicators suggest average performance. Increasing study time and maintaining attendance may help. 💪";

    }

    else {

        message.textContent =
            "The current indicators suggest that improvement is needed. Focus on attendance, study consistency and assessments. 🎯";
    }


    // ===============================
    // RESULT COLOR
    // ===============================

    const predictionElement =
        document.getElementById("prediction");

    if (prediction === "Excellent") {

        predictionElement.style.color = "#00e676";

    }

    else if (prediction === "Good") {

        predictionElement.style.color = "#29b6f6";

    }

    else if (prediction === "Average") {

        predictionElement.style.color = "#ffca28";

    }

    else {

        predictionElement.style.color = "#ff5252";
    }


    // ===============================
    // CONSOLE INFORMATION
    // ===============================

    console.log("Prediction:", prediction);
    console.log("Confidence:", confidence.toFixed(2) + "%");
    console.log("Python model accuracy:", model.accuracy + "%");
}


// ===============================
// LOAD MODEL WHEN PAGE OPENS
// ===============================

loadModel();