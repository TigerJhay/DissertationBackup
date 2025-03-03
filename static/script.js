const dropArea = document.getElementById('cust_drop-area-container');
const imagePreview = document.getElementById('cust_image-preview');
const urlDisplay = document.getElementById('cust_url-display');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
  dropArea.style.borderColor = '#007bff';
}

function unhighlight(e) {
  dropArea.style.borderColor = '#ccc';
}

// Handle dropped data
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  e.preventDefault();
  const data = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');

  if (data) {
    if (isValidImageUrl(data)) {
      // imagePreview.src = data;
      // imagePreview.style.display = 'block';
      imagePreview.style.backgroundImage =`url(${data})`
      urlDisplay.textContent = `Image URL: ${data}`;
      
    } else {
      urlDisplay.textContent = 'Invalid Image URL';
      // imagePreview.style.display = 'none';
    }
  }
}

function isValidImageUrl(url) {
  try {
    const parsedUrl = new URL(url);
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']; // Add more if needed
    const lowerCaseUrl = parsedUrl.pathname.toLowerCase();
    return imageExtensions.some(extension => lowerCaseUrl.endsWith(extension));
  } catch (_) {
    return false;
  }
}