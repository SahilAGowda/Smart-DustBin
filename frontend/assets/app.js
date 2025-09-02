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

// On dashboard page
const fileInput = document.getElementById("fileInput");
const scanBtn = document.getElementById("scanBtn");
const previewDiv = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const objectLabel = document.getElementById("objectLabel");
const wasteType = document.getElementById("wasteType");
const confidence = document.getElementById("confidence");
const recycleBar = document.getElementById("recycleBar");
const modelStatus = document.getElementById("modelStatus");
const deviceInfo = document.getElementById("deviceInfo");

const logoutLink = document.getElementById("logoutLink");
if (logoutLink) {
  logoutLink.addEventListener("click", (e) => {
    e.preventDefault();
    clearToken();
    window.location.href = "index.html";
  });
}

if (fileInput) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    previewDiv.innerHTML = `<img src="${url}" alt="preview" />`;
  });
}

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
      objectLabel.textContent = data.object_label;
      wasteType.textContent = data.waste_type;
      confidence.textContent = String(data.confidence);
      recycleBar.style.width = `${data.recyclability_percent}%`;
      
      // Update model info
      if (modelStatus && data.model_status) {
        modelStatus.textContent = data.model_status === "loaded" ? "CNN Model" : "Mock Model";
      }
      if (deviceInfo && data.device) {
        deviceInfo.textContent = data.device;
      }
      
      resultDiv.classList.remove("hidden");
    } catch (e) {
      alert("Network error");
    }
  });
}
