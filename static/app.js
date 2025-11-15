document.addEventListener("DOMContentLoaded", () => {
    const predictBtn = document.getElementById("predictBtn");
    const imageInput = document.getElementById("imageInput");
    const loading = document.getElementById("loading");
    const fileNameSpan = document.getElementById("fileName");
    const resultPanel = document.getElementById("result");
    
    // Image Preview Elements
    const imagePreview = document.getElementById("xrayImagePreview");
    const previewPlaceholder = document.getElementById("previewPlaceholder");

    // Update file name and show preview
    imageInput.addEventListener("change", (event) => {
        if (event.target.files.length > 0) {
            const file = event.target.files[0];
            fileNameSpan.textContent = file.name;

            // Use FileReader to show image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
            };
            reader.readAsDataURL(file);

        } else {
            fileNameSpan.textContent = "No file selected";
            imagePreview.src = ""; // Clear preview
        }
    });

    predictBtn.addEventListener("click", async () => {
        if (!imageInput.files.length) {
            alert("Please select an image!");
            return;
        }

        // Hide results, show loading spinner
        loading.classList.remove("hidden");
        resultPanel.classList.add("hidden");

        const file = imageInput.files[0];
        const formData = new FormData();
        formData.append("file", file);
        formData.append("use_llm", "true");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "x-api-key": "dev-key" }, // Make sure this key is correct
                body: formData
            });

            if (!response.ok) {
                let errText;
                try { errText = await response.text(); } catch (e) { errText = response.statusText; }
                alert(`Server error: ${response.status} ${response.statusText}\n${errText}`);
                console.error('Non-OK response', response.status, errText);
                return;
            }

            const data = await response.json();
            console.log("/predict response:", data);

            // --- Populate Original Hidden Fields (for compatibility) ---
            const labelEl = document.getElementById("label");
            const scoreEl = document.getElementById("score");
            labelEl.innerText = data?.label ?? "N/A";
            scoreEl.innerText = (data?.score ?? "N/A").toString();

            // --- Populate New UI Elements ---

            // 1. Donut Chart
            const chartScoreEl = document.getElementById("chart-score");
            const chartLabelEl = document.getElementById("chart-label");
            const chartContainer = document.querySelector(".prediction-chart");

            let mainScore = 0;
            if (data && typeof data.score === 'number' && isFinite(data.score)) {
                mainScore = data.score;
            } else if (data && data.score != null) {
                const n = Number(data.score);
                mainScore = isFinite(n) ? n : 0;
            }
            
            const mainPercentage = (mainScore * 100);
            chartScoreEl.innerText = `${mainPercentage.toFixed(1)}%`;
            chartLabelEl.innerText = data?.label ?? "N/A";

            // Update conic-gradient for the chart
            chartContainer.style.background = `conic-gradient(
                var(--accent-chart) 0% ${mainPercentage}%,
                var(--bg-dark-primary) ${mainPercentage}% 100%
            )`;
            chartContainer.setAttribute("aria-valuenow", mainPercentage.toFixed(1));

            // 2. All Probabilities (Bars)
            const allScoresUl = document.getElementById("allScores");
            allScoresUl.innerHTML = ""; // Clear previous
            
            if (data && data.all_scores && typeof data.all_scores === 'object') {
                // Sort scores from high to low
                const sortedScores = Object.entries(data.all_scores)
                    .sort(([, v1], [, v2]) => v2 - v1);

                for (const [k, v] of sortedScores) {
                    let scoreValue = 0;
                    if (typeof v === 'number' && isFinite(v)) {
                        scoreValue = v;
                    } else {
                        const num = Number(v);
                        scoreValue = isFinite(num) ? num : 0;
                    }
                    
                    const percentage = (scoreValue * 100).toFixed(1);
                    const li = document.createElement("li");
                    li.innerHTML = `
                        <span class="label" title="${k}">${k}</span>
                        <div class="bar-container">
                            <div class="bar" style="width: ${percentage}%"></div>
                        </div>
                        <span class="percentage">${percentage}%</span>
                    `;
                    allScoresUl.appendChild(li);
                }
            } else {
                const li = document.createElement("li");
                li.textContent = "No other scores returned";
                allScoresUl.appendChild(li);
            }

            // 3. LLM Explanation
            const explanationEl = document.getElementById("explanation");
            explanationEl.innerText = (data && data.explanation) ? data.explanation : "No explanation available.";

            // --- Show results ---
            resultPanel.classList.remove("hidden");

        } catch (err) {
            alert("Error: " + (err && err.message ? err.message : String(err)));
            console.error(err);
        } finally {
            // Hide loading spinner
            loading.classList.add("hidden");
        }
    });
});