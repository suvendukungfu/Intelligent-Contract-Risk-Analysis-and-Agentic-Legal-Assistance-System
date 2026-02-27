document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        showError('Please select a PDF or TXT file to analyze.');
        return;
    }

    // UI State transitions
    document.getElementById('onboardingState').classList.add('hidden');
    document.getElementById('resultsState').classList.add('hidden');
    document.getElementById('errorState').classList.add('hidden');
    document.getElementById('loadingState').classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        document.getElementById('loadingState').classList.add('hidden');

        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze document.');
        }

        renderResults(data.summary, data.results);
        
    } catch (err) {
        document.getElementById('loadingState').classList.add('hidden');
        showError(err.message);
    }
});

function showError(message) {
    const errBox = document.getElementById('errorState');
    errBox.textContent = message;
    errBox.classList.remove('hidden');
}

function renderResults(summary, clauses) {
    document.getElementById('resultsState').classList.remove('hidden');

    // Calculate metrics
    const total = clauses.length;
    const highRisk = clauses.filter(c => c['Risk Level'] === 'High Risk').length;
    const exposure = total > 0 ? (highRisk / total * 100).toFixed(1) : 0;
    
    let totalConfidence = 0;
    clauses.forEach(c => totalConfidence += c.Confidence);
    const meanConf = total > 0 ? ((totalConfidence / total) * 100).toFixed(1) : 0;

    // Update KPIs
    document.getElementById('kpiTotal').textContent = total;
    document.getElementById('kpiHighRisk').textContent = highRisk;
    document.getElementById('kpiExposure').textContent = `${exposure}%`;
    document.getElementById('kpiConfidence').textContent = `${meanConf}%`;

    // Render Clauses
    const container = document.getElementById('clausesContainer');
    container.innerHTML = ''; // Clear previous

    // Sort by confidence (highest first)
    const sortedFields = [...clauses].sort((a, b) => b.Confidence - a.Confidence);

    sortedFields.forEach(clause => {
        const isHigh = clause['Risk Level'] === 'High Risk';
        const confPercent = (clause.Confidence * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'clause-item';

        const badgeClass = isHigh ? 'badge-high' : 'badge-low';
        const badgeText = isHigh ? 'HIGH RISK' : 'LOW RISK';
        const assessClass = isHigh ? 'assessment-high' : 'assessment-low';
        const msg = isHigh 
            ? 'Significant legal exposure detected. Linguistic triggers indicate elevated liability or obligation profile.'
            : 'Standard operational language. Low liability signature detected.';

        let chipsHtml = '';
        if (clause.Keywords && clause.Keywords.length > 0) {
            clause.Keywords.slice(0, 12).forEach(kw => {
                chipsHtml += `<span class="kw-chip">${kw}</span>`;
            });
        }

        item.innerHTML = `
            <div class="clause-header">
                <div>
                    <span class="badge ${badgeClass}">${badgeText}</span>
                    <span style="font-size: 0.8rem; color: var(--color-muted); margin-left:10px;">Segment ${clause.id + 1} · Confidence: ${confPercent}%</span>
                </div>
                <div style="color:var(--color-muted); font-size: 0.8rem;">▼</div>
            </div>
            <div class="clause-content">
                <p style="font-size:0.78rem;font-weight:600;color:var(--color-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Provision Text</p>
                <div class="clause-text">${clause.Clause}</div>
                
                <div class="assessment-box ${assessClass}">
                    <p class="assessment-title">Model Assessment</p>
                    <p class="assessment-msg">${msg}</p>
                </div>

                ${chipsHtml ? `
                <div style="margin-top:12px;">
                    <p style="font-size:0.78rem;font-weight:600;color:var(--color-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Linguistic Triggers</p>
                    ${chipsHtml}
                </div>
                ` : ''}
            </div>
        `;

        // Simple toggle logic
        const content = item.querySelector('.clause-content');
        if (!isHigh) content.classList.add('hidden'); // Collapse low risk by default

        item.querySelector('.clause-header').addEventListener('click', () => {
            content.classList.toggle('hidden');
        });

        container.appendChild(item);
    });
}
