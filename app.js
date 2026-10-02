const API_URL = "http://localhost:8000";

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const imgPreview = document.getElementById("image-preview");
const analyzeBtn = document.getElementById("analyze-btn");
const loadingDiv = document.getElementById("loading");
const resultsDiv = document.getElementById("results");
const reportBox = document.getElementById("ai-report-text");

// File Handling
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = () => {
            imgPreview.src = reader.result;
            previewContainer.style.display = "block";
            dropZone.style.display = "none";
            resultsDiv.style.display = "none";
        };
        reader.readAsDataURL(file);
    }
});

// Analysis
analyzeBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    analyzeBtn.disabled = true;
    loadingDiv.style.display = "block";
    resultsDiv.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        alert("Analysis failed. Backend might be offline. Check console.");
        console.error(error);
    } finally {
        loadingDiv.style.display = "none";
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    // 1. Prediction Card
    document.getElementById("pred-class-name").innerText = data.predicted_class;

    // Risk Badge
    const badge = document.getElementById("risk-badge");
    badge.innerText = data.risk_level;
    badge.className = "risk-badge";
    if (data.risk_level.includes("High")) badge.classList.add("risk-high");
    else if (data.risk_level.includes("Medium")) badge.classList.add("risk-med");
    else badge.classList.add("risk-low");

    // Confidence
    document.getElementById("confidence-val").innerText = (data.confidence * 100).toFixed(1) + "%";
    document.getElementById("conf-bar-fill").style.width = (data.confidence * 100) + "%";

    // 2. Chart
    renderChart(data.probabilities);

    // 3. Concepts
    const conceptsDiv = document.getElementById("concepts-list");
    conceptsDiv.innerHTML = "";
    // Display concepts
    if (data.concept_scores) {
        for (const [key, val] of Object.entries(data.concept_scores)) {
            const item = document.createElement("div");
            item.className = "concept-item";
            // Check if val has structure
            const level = val.level || "Unknown";
            const score = val.score || 0;

            item.innerHTML = `
                <span style="text-transform: capitalize">${key}</span>
                <span class="concept-val level-${level}">${level} <small>(${score.toFixed(2)})</small></span>
            `;
            conceptsDiv.appendChild(item);
        }
    }

    // 4. Explainability Images
    document.getElementById("roi-img").src = data.explainability.roi;
    document.getElementById("gradcam-img").src = data.explainability.gradcam;
    document.getElementById("ig-img").src = data.explainability.integrated_gradients;

    // 5. Report
    reportBox.innerText = data.ai_explanation;

    // Show
    resultsDiv.style.display = "block";
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

let chartInstance = null;

function renderChart(probs) {
    const ctx = document.getElementById('probChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    const labels = Object.keys(probs);
    const values = Object.values(probs).map(p => p * 100);
    const backgroundColors = labels.map(l => {
        if (['mel', 'bcc', 'akiec'].includes(l)) return '#ef4444';
        return '#10b981';
    });

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: backgroundColors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { x: { beginAtZero: true, max: 100 } }
        }
    });
}
