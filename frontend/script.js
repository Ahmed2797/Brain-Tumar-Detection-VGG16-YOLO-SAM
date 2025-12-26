document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    const selectedFile = document.getElementById('selectedFile');
    const predictVggBtn = document.getElementById('predictVggBtn');
    const detectYoloBtn = document.getElementById('detectYoloBtn');
    const segmentSamBtn = document.getElementById('segmentSamBtn');
    const resetBtn = document.getElementById('resetBtn');
    const uploadedImage = document.getElementById('uploadedImage');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const segmentationCanvas = document.getElementById('segmentationCanvas');
    const apiStatus = document.getElementById('apiStatus');
    const apiStatusText = document.getElementById('apiStatusText');
    const predictionLabel = document.getElementById('predictionLabel');
    const predictionConfidence = document.getElementById('predictionConfidence');
    const confidenceFill = document.getElementById('confidenceFill');
    const predictionDetails = document.querySelector('#predictionResult .result-details');
    const predictionPlaceholder = document.querySelector('#predictionResult .result-placeholder');
    const detectionDetails = document.querySelector('#detectionResult .result-details');
    const detectionPlaceholder = document.querySelector('#detectionResult .result-placeholder');
    const detectionImageContainer = document.getElementById('detectionImageContainer');
    const segmentationDetails = document.querySelector('#segmentationResult .result-details');
    const segmentationPlaceholder = document.querySelector('#segmentationResult .result-placeholder');
    const segmentationImageContainer = document.getElementById('segmentationImageContainer');
    
    // State variables
    let currentImage = null;
    
    // Check API status on page load
    checkApiStatus();
    
    // Event Listeners
    imageInput.addEventListener('change', handleImageUpload);
    
    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.backgroundColor = '#f0f3ff';
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.backgroundColor = '';
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.backgroundColor = '';
        if (e.dataTransfer.files.length) {
            imageInput.files = e.dataTransfer.files;
            handleImageUpload();
        }
    });
    
    predictVggBtn.addEventListener('click', handlePredictVgg);
    detectYoloBtn.addEventListener('click', handleDetectYolo);
    segmentSamBtn.addEventListener('click', handleSegmentSam);
    resetBtn.addEventListener('click', handleReset);
    
    // Functions
    function checkApiStatus() {
        fetch('/api')
            .then(response => {
                if (response.ok) {
                    updateApiStatus(true, 'API connected');
                } else {
                    updateApiStatus(false, 'API connection failed');
                }
            })
            .catch(error => {
                console.error('API check failed:', error);
                updateApiStatus(false, 'Cannot connect to API');
            });
    }
    
    function updateApiStatus(isConnected, message) {
        if (isConnected) {
            apiStatus.className = 'status-dot connected';
            apiStatusText.textContent = message;
        } else {
            apiStatus.className = 'status-dot';
            apiStatusText.textContent = message;
        }
    }
    
    function handleImageUpload() {
        const file = imageInput.files[0];
        if (!file) return;
        
        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (PNG, JPG, JPEG, BMP, TIFF)');
            return;
        }
        
        selectedFile.textContent = file.name;
        currentImage = file;
        
        // Display the image
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImage.classList.remove('hidden');
            imagePlaceholder.classList.add('hidden');
            
            // Reset canvas
            segmentationCanvas.classList.add('hidden');
            segmentationCanvas.width = uploadedImage.naturalWidth;
            segmentationCanvas.height = uploadedImage.naturalHeight;
            
            // Enable buttons
            predictVggBtn.disabled = false;
            detectYoloBtn.disabled = false;
            segmentSamBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    async function handlePredictVgg() {
        if (!currentImage) {
            alert('Please upload an image first');
            return;
        }
        
        // Disable buttons during processing
        predictVggBtn.disabled = true;
        predictVggBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        const formData = new FormData();
        formData.append('file', currentImage);
        
        try {
            const response = await fetch('/predict_vgg', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Display results
            displayPredictionResult(result);
            
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Prediction failed. Please try again.');
        } finally {
            predictVggBtn.disabled = false;
            predictVggBtn.innerHTML = '<i class="fas fa-search"></i> Predict VGG';
        }
    }
    
    async function handleDetectYolo() {
        if (!currentImage) {
            alert('Please upload an image first');
            return;
        }
        
        // Disable buttons during processing
        detectYoloBtn.disabled = true;
        detectYoloBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        const formData = new FormData();
        formData.append('file', currentImage);
        
        try {
            const response = await fetch('/detect_yolo', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            // Get the image blob
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            
            // Display the annotated image
            displayAnnotatedImage(detectionImageContainer, imageUrl);
            
        } catch (error) {
            console.error('Detection error:', error);
            alert('Detection failed. Please try again.');
        } finally {
            detectYoloBtn.disabled = false;
            detectYoloBtn.innerHTML = '<i class="fas fa-crosshairs"></i> Detect YOLO';
        }
    }
    
    async function handleSegmentSam() {
        if (!currentImage) {
            alert('Please upload an image first');
            return;
        }
        
        // Disable buttons during processing
        segmentSamBtn.disabled = true;
        segmentSamBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        const formData = new FormData();
        formData.append('file', currentImage);
        
        try {
            const response = await fetch('/segment_sam', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            // Get the image blob
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            
            // Display the annotated image
            displayAnnotatedImage(segmentationImageContainer, imageUrl);
            
        } catch (error) {
            console.error('Segmentation error:', error);
            alert('Segmentation failed. Please try again.');
        } finally {
            segmentSamBtn.disabled = false;
            segmentSamBtn.innerHTML = '<i class="fas fa-draw-polygon"></i> Segment SAM';
        }
    }
    
    function displayPredictionResult(result) {
        const { prediction, confidence } = result;
        const confidencePercent = (confidence * 100).toFixed(2);
        
        // Update UI
        predictionLabel.textContent = prediction;
        predictionConfidence.textContent = `${confidencePercent}%`;
        
        // Animate confidence bar
        confidenceFill.style.width = `${confidencePercent}%`;
        
        // Color code based on confidence
        if (confidence > 0.8) {
            confidenceFill.style.background = 'linear-gradient(to right, #ff6b6b, #ff4757)';
        } else if (confidence > 0.5) {
            confidenceFill.style.background = 'linear-gradient(to right, #ffa502, #ff7f00)';
        } else {
            confidenceFill.style.background = 'linear-gradient(to right, #2ed573, #1dd1a1)';
        }
        
        // Show result details
        predictionPlaceholder.classList.add('hidden');
        predictionDetails.classList.remove('hidden');
    }
    

    

    

    
    function handleReset() {
        // Reset all UI elements
        imageInput.value = '';
        selectedFile.textContent = 'No file selected';
        uploadedImage.src = '';
        uploadedImage.classList.add('hidden');
        segmentationCanvas.classList.add('hidden');
        imagePlaceholder.classList.remove('hidden');
        
        predictionPlaceholder.classList.remove('hidden');
        predictionDetails.classList.add('hidden');
        
        detectionPlaceholder.classList.remove('hidden');
        detectionDetails.classList.add('hidden');
        detectionImageContainer.innerHTML = '';
        
        segmentationPlaceholder.classList.remove('hidden');
        segmentationDetails.classList.add('hidden');
        segmentationImageContainer.innerHTML = '';
        
        confidenceFill.style.width = '0%';
        
        predictVggBtn.disabled = true;
        detectYoloBtn.disabled = true;
        segmentSamBtn.disabled = true;
        
        currentImage = null;
    }
    
    function displayAnnotatedImage(container, imageUrl) {
        // Clear previous content
        container.innerHTML = '';
        
        const img = document.createElement('img');
        img.src = imageUrl;
        img.className = 'annotated-image';
        img.alt = 'Annotated Image';
        
        container.appendChild(img);
        
        // Show result details
        if (container === detectionImageContainer) {
            detectionPlaceholder.classList.add('hidden');
            detectionDetails.classList.remove('hidden');
        } else if (container === segmentationImageContainer) {
            segmentationPlaceholder.classList.add('hidden');
            segmentationDetails.classList.remove('hidden');
        }
    }
    
    // Initialize button states
    predictVggBtn.disabled = true;
    detectYoloBtn.disabled = true;
    segmentSamBtn.disabled = true;
});