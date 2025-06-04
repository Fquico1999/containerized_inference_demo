const modelSelect = document.getElementById('model_name');
const customModelInput = document.getElementById('custom_model_name');
const taskSelect = document.getElementById('task');
const customTaskInput = document.getElementById('custom_task'); // Assuming you might want custom task input
const inputsTextarea = document.getElementById('inputs');
const pipelineKwargsInput = document.getElementById('pipeline_kwargs');
const responseOutput = document.getElementById('responseOutput');
const loader = document.getElementById('loader');

// Pre-fill task based on selected model (basic examples)
const modelTaskMap = {
    "distilbert-base-uncased-finetuned-sst-2-english": "sentiment-analysis",
    "gpt2": "text-generation",
    "facebook/bart-large-cnn": "summarization"
};

modelSelect.addEventListener('change', function() {
    if (this.value === 'custom') {
        customModelInput.style.display = 'block';
        customModelInput.value = ''; // Clear previous custom model
    } else {
        customModelInput.style.display = 'none';
        if (modelTaskMap[this.value]) {
            taskSelect.value = modelTaskMap[this.value];
        }
    }
    // Could also add logic for custom_task if 'custom' is selected for task
});


async function submitInference() {
    loader.style.display = 'block';
    responseOutput.textContent = 'Loading...';

    let modelName = modelSelect.value;
    if (modelName === 'custom') {
        modelName = customModelInput.value.trim();
        if (!modelName) {
            alert("Please enter a custom model name.");
            loader.style.display = 'none';
            responseOutput.textContent = 'Error: Custom model name required.';
            return;
        }
    }

    let task = taskSelect.value;
    // Add logic for customTaskInput if you implement it fully

    const rawInputs = inputsTextarea.value.trim();
    let processedInputs;

    if (!rawInputs) {
        alert("Please enter input text.");
        loader.style.display = 'none';
        responseOutput.textContent = 'Error: Input text required.';
        return;
    }

    if (rawInputs.includes('\n')) { // Batch
        processedInputs = rawInputs.split('\n').map(line => line.trim()).filter(line => line);
    } else { // Single input
        processedInputs = rawInputs;
    }


    let pipelineKwargs = {};
    if (pipelineKwargsInput.value.trim()) {
        try {
            pipelineKwargs = JSON.parse(pipelineKwargsInput.value.trim());
        } catch (e) {
            alert("Invalid JSON in Pipeline Kwargs.");
            loader.style.display = 'none';
            responseOutput.textContent = 'Error: Invalid JSON in Pipeline Kwargs.';
            return;
        }
    }

    const payload = {
        model_name: modelName,
        task: task,
        inputs: processedInputs,
        pipeline_kwargs: pipelineKwargs
    };

    try {
        // The UI is served from the root, so API calls are relative
        const response = await fetch('/api/predict', { // Nginx will proxy /api/predict
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const responseData = await response.json();
        
        if (!response.ok) {
            // Try to get detail from FastAPI error response
            const errorDetail = responseData.detail || JSON.stringify(responseData);
            responseOutput.textContent = `Error ${response.status}: ${errorDetail}`;
        } else {
            responseOutput.textContent = JSON.stringify(responseData, null, 2);
        }

    } catch (error) {
        console.error("Fetch error:", error);
        responseOutput.textContent = 'Network error or API unreachable. Check console. ' + error.message;
    } finally {
        loader.style.display = 'none';
    }
}