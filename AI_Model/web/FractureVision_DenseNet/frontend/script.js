document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const predictBtn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('resultsSection');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const loader = document.getElementById('loader');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImg = document.getElementById('removeImg');

    const bboxImg = document.getElementById('bboxImg');
    const maskImg = document.getElementById('maskImg');
    const probValue = document.getElementById('probValue');
    const statusBadge = document.getElementById('statusBadge');

    let selectedFile = null;

    // --- Drag & Drop ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'));
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            dropZone.style.display = 'none';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeImg.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        dropZone.style.display = 'flex';
        predictBtn.disabled = true;
        resultsSection.style.display = 'none';
        resultsPlaceholder.style.display = 'block';
    });

    // --- Prediction Logic ---
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Reset UI
        resultsSection.style.display = 'none';
        resultsPlaceholder.style.display = 'none';
        loader.style.display = 'block';
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis failed');

            const data = await response.json();

            // Update Results
            bboxImg.src = `data:image/png;base64,${data.image_bbox}`;
            
            if (data.has_mask) {
                maskImg.src = `data:image/png;base64,${data.image_mask}`;
                maskImg.parentElement.style.display = 'block';
            } else {
                // If no mask, we can either hide it or show the highlighted original
                maskImg.src = `data:image/png;base64,${data.image_mask}`;
                maskImg.nextElementSibling.textContent = "AI Analysis (Highlight)";
            }

            probValue.textContent = `${data.probability}%`;
            
            const isFrac = data.status === 'Fractured';
            statusBadge.textContent = isFrac ? 'FRACTURE DETECTED' : 'NO FRACTURE';
            statusBadge.className = `status-badge ${isFrac ? 'status-fractured' : 'status-normal'}`;

            // Show results
            loader.style.display = 'none';
            resultsSection.style.display = 'flex';
        } catch (error) {
            console.error(error);
            alert('Error connecting to AI Server. Please ensure the backend is running.');
            loader.style.display = 'none';
            resultsPlaceholder.style.display = 'block';
            predictBtn.disabled = false;
        }
    });
});
