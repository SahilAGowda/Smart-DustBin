const API_BASE = "";

// Simple token handling for prototype
function setToken(t) { localStorage.setItem("sd_token", t); }
function getToken() { return localStorage.getItem("sd_token") || ""; }
function clearToken() { localStorage.removeItem("sd_token"); }

// On login page
const loginBtn = document.getElementById("loginBtn");
if (loginBtn) {
  loginBtn.addEventListener("click", async () => {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value.trim();
    const msg = document.getElementById("loginMsg");

    try {
      const res = await fetch(`${API_BASE}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (data.ok) {
        setToken(data.token);
        msg.textContent = "Logged in! Redirecting…";
        window.location.href = "dashboard.html";
      } else {
        msg.textContent = data.error || "Login failed";
      }
    } catch (e) {
      msg.textContent = "Network error";
    }
  });
}

// Dashboard page elements
const fileInput = document.getElementById("fileInput");
const scanBtn = document.getElementById("scanBtn");
const clearBtn = document.getElementById("clearBtn");
const uploadZone = document.getElementById("uploadZone");
const previewDiv = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const objectLabel = document.getElementById("objectLabel");
const wasteType = document.getElementById("wasteType");
const confidence = document.getElementById("confidence");
const confidenceDisplay = document.getElementById("confidenceDisplay");
const recycleBar = document.getElementById("recycleBar");
const recyclePercent = document.getElementById("recyclePercent");
const modelStatus = document.getElementById("modelStatus");
const deviceInfo = document.getElementById("deviceInfo");
const objectIcon = document.getElementById("objectIcon");
const wasteIcon = document.getElementById("wasteIcon");
const recyclingTips = document.getElementById("recyclingTips");
const scanAgainBtn = document.getElementById("scanAgainBtn");
const saveResultBtn = document.getElementById("saveResultBtn");

// Logout functionality
const logoutLink = document.getElementById("logoutLink");
if (logoutLink) {
  logoutLink.addEventListener("click", (e) => {
    e.preventDefault();
    clearToken();
    window.location.href = "index.html";
  });
}

// Drag and drop functionality
if (uploadZone) {
  uploadZone.addEventListener("click", () => {
    fileInput.click();
  });

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      handleFileSelect();
    }
  });
}

// File input change handler
if (fileInput) {
  fileInput.addEventListener("change", handleFileSelect);
}

function handleFileSelect() {
  const file = fileInput.files[0];
  if (!file) return;

  // Validate file type
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file.');
    return;
  }

  // Show preview
  const url = URL.createObjectURL(file);
  previewDiv.innerHTML = `
    <div style="text-align: center;">
      <img src="${url}" alt="preview" style="max-width: 100%; max-height: 400px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.3);" />
      <p style="margin-top: 1rem; color: var(--fg-secondary);">
        <i class="fas fa-image"></i> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
      </p>
    </div>
  `;

  // Enable scan button and show clear button
  scanBtn.disabled = false;
  clearBtn.style.display = 'inline-flex';
  
  // Hide previous results
  resultDiv.classList.add('hidden');
}

// Clear functionality
if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    fileInput.value = '';
    previewDiv.innerHTML = '';
    scanBtn.disabled = true;
    clearBtn.style.display = 'none';
    resultDiv.classList.add('hidden');
  });
}

// Scan functionality
if (scanBtn) {
  scanBtn.addEventListener("click", async () => {
    const token = getToken();
    if (!token) {
      alert("Please login again.");
      window.location.href = "index.html";
      return;
    }
    
    const file = fileInput.files[0];
    if (!file) {
      alert("Please choose an image first.");
      return;
    }

    // Show loading state
    scanBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    scanBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      });
      
      const data = await res.json();
      
      if (!data.ok) {
        alert(data.error || "Prediction failed");
        return;
      }

      // Update results with enhanced display
      displayResults(data);
      
    } catch (e) {
      alert("Network error: " + e.message);
    } finally {
      // Reset scan button
      scanBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Waste';
      scanBtn.disabled = false;
    }
  });
}

function displayResults(data) {
  // Update basic info
  objectLabel.textContent = formatObjectLabel(data.object_label);
  wasteType.textContent = formatWasteType(data.waste_type);
  confidence.textContent = data.confidence;
  
  // Update confidence display
  const confidencePercent = Math.round(data.confidence * 100);
  confidenceDisplay.textContent = `${confidencePercent}%`;
  
  // Update confidence badge color based on value
  const confidenceBadge = document.querySelector('.confidence-badge');
  if (confidencePercent >= 80) {
    confidenceBadge.style.background = 'var(--gradient-success)';
  } else if (confidencePercent >= 60) {
    confidenceBadge.style.background = 'var(--accent-warning)';
  } else {
    confidenceBadge.style.background = 'var(--accent-error)';
  }

  // Update recyclability bar
  recycleBar.style.width = `${data.recyclability_percent}%`;
  recyclePercent.textContent = `${data.recyclability_percent}%`;
  
  // Update bar color based on recyclability
  if (data.recyclability_percent >= 70) {
    recycleBar.style.background = 'var(--gradient-success)';
  } else if (data.recyclability_percent >= 40) {
    recycleBar.style.background = 'linear-gradient(135deg, var(--accent-warning), #f59e0b)';
  } else {
    recycleBar.style.background = 'linear-gradient(135deg, var(--accent-error), #dc2626)';
  }

  // Update icons
  updateIcons(data.object_label, data.waste_type);
  
  // Update model info
  if (modelStatus && data.model_status) {
    modelStatus.textContent = data.model_status === "loaded" ? "CNN Model" : "Mock Model";
  }
  if (deviceInfo && data.device) {
    deviceInfo.textContent = data.device;
  }

  // Update recycling tips
  updateRecyclingTips(data.waste_type, data.recyclability_percent);

  // Show results with animation
  resultDiv.classList.remove("hidden");
  resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function formatObjectLabel(label) {
  return label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatWasteType(type) {
  const typeMap = {
    'plastic': 'Plastic Waste',
    'metal': 'Metal Waste', 
    'paper': 'Paper Waste',
    'glass': 'Glass Waste',
    'organic': 'Organic Waste',
    'e-waste': 'Electronic Waste',
    'medical': 'Medical Waste',
    'other': 'General Waste'
  };
  return typeMap[type] || type.charAt(0).toUpperCase() + type.slice(1);
}

function updateIcons(objectLabel, wasteType) {
  // Object icons
  const objectIconMap = {
    'plastic_bottle': 'fas fa-wine-bottle',
    'aluminum_can': 'fas fa-beer',
    'cardboard': 'fas fa-box',
    'paper': 'fas fa-file-alt',
    'glass_jar': 'fas fa-glass-whiskey',
    'food_waste': 'fas fa-apple-alt',
    'old_phone': 'fas fa-mobile-alt',
    'laptop': 'fas fa-laptop',
    'syringe': 'fas fa-syringe',
    'battery': 'fas fa-battery-half',
    'other': 'fas fa-question'
  };

  // Waste type icons
  const wasteIconMap = {
    'plastic': 'fas fa-wine-bottle',
    'metal': 'fas fa-cog',
    'paper': 'fas fa-newspaper',
    'glass': 'fas fa-glass-whiskey',
    'organic': 'fas fa-leaf',
    'e-waste': 'fas fa-microchip',
    'medical': 'fas fa-plus-square',
    'other': 'fas fa-trash'
  };

  if (objectIcon) {
    objectIcon.className = objectIconMap[objectLabel] || 'fas fa-eye';
  }
  
  if (wasteIcon) {
    wasteIcon.className = wasteIconMap[wasteType] || 'fas fa-trash';
  }
}

function updateRecyclingTips(wasteType, recyclability) {
  const tips = {
    'plastic': {
      high: 'Great! Clean this plastic item and place it in your recycling bin. Remove any labels if possible.',
      medium: 'This plastic can be recycled at most facilities. Check the recycling number on the bottom.',
      low: 'Limited recycling options. Consider reusing or find specialized plastic recycling programs.'
    },
    'metal': {
      high: 'Excellent! Metals are highly recyclable. Clean and place in metal recycling.',
      medium: 'Good recyclability. Remove any non-metal parts before recycling.',
      low: 'Some metal components may be recyclable. Contact local recycling center.'
    },
    'paper': {
      high: 'Perfect for paper recycling! Keep it clean and dry.',
      medium: 'Recyclable if clean. Remove any plastic components first.',
      low: 'Limited recycling due to coatings or contamination.'
    },
    'glass': {
      high: 'Excellent! Glass is infinitely recyclable. Clean and sort by color if required.',
      medium: 'Good for recycling. Remove caps and labels.',
      low: 'Check local programs for specialty glass recycling.'
    },
    'organic': {
      high: 'Perfect for composting! Create nutrient-rich soil.',
      medium: 'Compostable in most home systems.',
      low: 'Not recyclable, but can be composted to reduce waste.'
    },
    'e-waste': {
      high: 'Take to certified e-waste recycling facility for proper processing.',
      medium: 'Contains recoverable materials. Use e-waste collection programs.',
      low: 'Requires specialized handling. Never put in regular trash.'
    },
    'medical': {
      high: 'Use medical waste disposal programs. Never put in regular recycling.',
      medium: 'Requires special handling. Contact healthcare providers.',
      low: 'Hazardous waste. Use designated medical waste disposal.'
    },
    'other': {
      high: 'Check local recycling guidelines for specific handling.',
      medium: 'May have recycling options. Research local programs.',
      low: 'Consider reuse options or proper disposal methods.'
    }
  };

  let level;
  if (recyclability >= 70) level = 'high';
  else if (recyclability >= 40) level = 'medium';
  else level = 'low';

  const tip = tips[wasteType]?.[level] || 'Check with your local waste management for disposal guidelines.';
  
  if (recyclingTips) {
    recyclingTips.innerHTML = `
      <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
        <i class="fas fa-lightbulb" style="color: var(--accent-warning); margin-top: 0.25rem;"></i>
        <p style="margin: 0; color: var(--fg-secondary); line-height: 1.6;">${tip}</p>
      </div>
    `;
  }
}

// Scan again functionality
if (scanAgainBtn) {
  scanAgainBtn.addEventListener("click", () => {
    fileInput.value = '';
    previewDiv.innerHTML = '';
    scanBtn.disabled = true;
    clearBtn.style.display = 'none';
    resultDiv.classList.add('hidden');
    
    // Scroll back to upload area
    uploadZone.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
}

// Save results functionality (placeholder)
if (saveResultBtn) {
  saveResultBtn.addEventListener("click", () => {
    const results = {
      timestamp: new Date().toISOString(),
      object: objectLabel.textContent,
      wasteType: wasteType.textContent,
      confidence: confidence.textContent,
      recyclability: recycleBar.style.width
    };
    
    // For now, just show the results
    alert(`Results saved!\n\nObject: ${results.object}\nWaste Type: ${results.wasteType}\nConfidence: ${results.confidence}\nRecyclability: ${results.recyclability}`);
    
    // In a real app, you'd send this to a backend API
  });
}
