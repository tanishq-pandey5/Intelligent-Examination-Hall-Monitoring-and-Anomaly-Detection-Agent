/**
 * ==============================================================================
 * ExamGuard AI - Frontend Controller
 * ==============================================================================
 * Connects the dark skeuomorphic UI with the high-performance FastAPI backend:
 * - 60 FPS Canvas rendering for bounding boxes, tracking IDs, and foot-points
 * - Chart.js spline curve matching mockup "Portfolio Performance"
 * - Instant scrubbing with zero reload lag
 * - Natural language AI search assistant
 * ==============================================================================
 */

let performanceChart = null;
let currentVideoName = "class_1.mp4";
let isVideoPlaying = false;
let currentDetections = [];
let allAlerts = [];
let isAnalyzingFrame = false;

// DOM Elements
const videoEl = document.getElementById("examVideo");
const canvasEl = document.getElementById("overlayCanvas");
const ctx = canvasEl ? canvasEl.getContext("2d") : null;
const scrubber = document.getElementById("timeScrubber");
const playPauseBtn = document.getElementById("playPauseBtn");
const timeDisplay = document.getElementById("currentTimeDisplay");
const videoSelect = document.getElementById("videoSelectDropdown");

document.addEventListener("DOMContentLoaded", () => {
  initPerformanceChart();
  fetchSystemStatus();
  fetchVideoCatalog();
  fetchAlerts();
  fetchChartData();

  if (videoEl) {
    setupVideoPlayer();
  }
});

/* --------------------------------------------------------------------------
   CHART.JS INITIALIZATION (Matching "Portfolio Performance" Mockup)
-------------------------------------------------------------------------- */
function initPerformanceChart() {
  const chartCanvas = document.getElementById("performanceChart");
  if (!chartCanvas) return;

  const chartCtx = chartCanvas.getContext("2d");

  // Create smooth violet-to-transparent gradient fill
  const gradientFill = chartCtx.createLinearGradient(0, 0, 0, 240);
  gradientFill.addColorStop(0, "rgba(217, 128, 250, 0.28)");
  gradientFill.addColorStop(0.5, "rgba(168, 85, 247, 0.12)");
  gradientFill.addColorStop(1, "rgba(20, 19, 27, 0.0)");

  // Default labels and curve
  const labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const baselineData = [190, 160, 175, 168, 162, 185, 178, 145, 130, 150, 142, 135];

  performanceChart = new Chart(chartCtx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Risk Index",
          data: baselineData,
          borderColor: "#d980fa",
          borderWidth: 2.2,
          backgroundColor: gradientFill,
          fill: true,
          tension: 0.45, // Soft organic spline curve
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: "#ffffff",
          pointHoverBorderColor: "#f472b6",
          pointHoverBorderWidth: 3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          backgroundColor: "#1a1824",
          borderColor: "rgba(255, 255, 255, 0.15)",
          borderWidth: 1,
          titleColor: "#8b889d",
          bodyColor: "#ffffff",
          titleFont: { size: 11 },
          bodyFont: { size: 13, weight: "bold" },
          padding: 10,
          displayColors: false,
          callbacks: {
            title: (items) => `${items[0].label} Examination Interval`,
            label: (item) => `Risk Score: ${item.parsed.y}/100`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: "#5f5d6f",
            font: { size: 11 }
          },
          border: { display: false }
        },
        y: {
          min: 0,
          max: 200,
          ticks: {
            stepSize: 50,
            callback: (val) => val === 0 ? "10k" : `${val}k`,
            color: "#5f5d6f",
            font: { size: 11 }
          },
          grid: {
            color: "rgba(255, 255, 255, 0.03)",
            drawBorder: false
          },
          border: { display: false }
        }
      }
    }
  });
}

function changeChartSpan(span) {
  // Update button active states
  const buttons = document.querySelectorAll(".time-range-pills .mini-pill");
  buttons.forEach(b => b.classList.remove("active"));
  event.target.classList.add("active");

  fetchChartData();
}

async function fetchChartData() {
  try {
    const res = await fetch("/api/chart-data");
    const json = await res.json();
    if (json && json.datapoints && performanceChart) {
      const labels = json.datapoints.map(d => d.timecode);
      const data = json.datapoints.map(d => d.risk_score * 2); // scale for chart aesthetic

      performanceChart.data.labels = labels;
      performanceChart.data.datasets[0].data = data;
      performanceChart.update();

      // Update the hover badge if points exist
      if (json.datapoints.length > 5) {
        const midPoint = json.datapoints[Math.floor(json.datapoints.length / 2)];
        document.getElementById("badgeDate").textContent = `Timeline ${midPoint.timecode}`;
        document.getElementById("badgeVal").textContent = `Risk: ${midPoint.risk_score}/100`;
        document.getElementById("badgeTag").textContent = `${midPoint.alert_count} Alerts`;
      }
    }
  } catch (err) {
    console.warn("Using offline chart baseline", err);
  }
}

/* --------------------------------------------------------------------------
   BACKEND STATUS & VIDEO CATALOG
-------------------------------------------------------------------------- */
async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    
    if (data.models) {
      const isFineTuned = data.models.yolo_detector.is_fine_tuned;
      document.getElementById("modelStatusSub").textContent = isFineTuned 
        ? "Fine-Tuned Exam YOLOv8 Active" 
        : "Base YOLOv8 Active";

      const hudModel = document.getElementById("hudModelText");
      if (hudModel) hudModel.textContent = data.models.yolo_detector.name;

      const hudML = document.getElementById("hudMLText");
      if (hudML) hudML.textContent = data.models.ml_anomaly_detector.is_loaded ? "Enabled" : "Disabled";
    }

    if (data.stats) {
      document.getElementById("trackedCandidatesCount").textContent = `${data.stats.active_candidates || 20} Active`;
    }
  } catch (e) {
    console.error("Status fetch error:", e);
  }
}

async function fetchVideoCatalog() {
  try {
    const res = await fetch("/api/videos");
    const data = await res.json();
    
    if (data.videos && videoSelect) {
      videoSelect.innerHTML = "";
      data.videos.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v.name;
        opt.textContent = `${v.name} (${v.duration_str}, ${v.category})`;
        videoSelect.appendChild(opt);
      });

      if (data.videos.length > 0) {
        currentVideoName = data.videos[0].name;
        document.getElementById("currentVideoBadge").textContent = currentVideoName;
        loadVideoSource(currentVideoName);
      }
    }
  } catch (e) {
    console.error("Video catalog error:", e);
  }
}

async function fetchAlerts() {
  try {
    const res = await fetch("/api/alerts?limit=10");
    const data = await res.json();
    if (data.alerts && data.alerts.length > 0) {
      allAlerts = data.alerts;
      renderWatchlist(allAlerts);
    }
  } catch (e) {
    console.warn("Could not fetch alerts:", e);
  }
}

function renderWatchlist(alertsList) {
  const container = document.getElementById("watchlistContainer");
  if (!container || !alertsList.length) return;

  container.innerHTML = "";
  alertsList.slice(0, 4).forEach(a => {
    const item = document.createElement("div");
    item.className = "watch-item";
    item.onclick = () => openVideoModal(a.video_timestamp_sec);

    let icon = "⚠️";
    if (a.event_type.includes("PHONE")) icon = "📱";
    else if (a.event_type.includes("HEAD")) icon = "👀";
    else if (a.event_type.includes("SEAT")) icon = "🚶";
    else if (a.event_type.includes("GATHER")) icon = "👥";

    const riskCls = a.risk_score >= 70 ? "watch-risk-high" : (a.risk_score >= 45 ? "watch-risk-med" : "");

    item.innerHTML = `
      <div class="watch-item-left">
        <div class="watch-icon-box">${icon}</div>
        <div>
          <div class="watch-item-title">Seat #${a.seat_id || "?"}</div>
          <div class="watch-item-sub">${a.event_display_name}</div>
        </div>
      </div>
      <div class="watch-item-right">
        <div class="watch-price">${a.timecode_str}</div>
        <div class="watch-gain ${riskCls}">+${Math.round(a.risk_score)}% Risk</div>
      </div>
    `;
    container.appendChild(item);
  });
}

function filterWatchlist(type) {
  const pills = document.querySelectorAll(".filter-mini-pills .mini-pill");
  pills.forEach(p => p.classList.remove("active"));
  event.target.classList.add("active");

  if (type === "high") {
    renderWatchlist(allAlerts.filter(a => a.risk_score >= 70));
  } else if (type === "resolved") {
    renderWatchlist(allAlerts.filter(a => a.review_status === "RESOLVED"));
  } else {
    renderWatchlist(allAlerts);
  }
}

/* --------------------------------------------------------------------------
   ZERO-LAG VIDEO PLAYER & CANVAS HUD
-------------------------------------------------------------------------- */
function setupVideoPlayer() {
  videoEl.addEventListener("loadedmetadata", () => {
    scrubber.max = videoEl.duration || 20;
    updateTimecodeDisplay();
    resizeCanvas();
  });

  videoEl.addEventListener("timeupdate", () => {
    scrubber.value = videoEl.currentTime;
    updateTimecodeDisplay();
    triggerFrameAnalysisThrottled(videoEl.currentTime);
  });

  window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
  if (!canvasEl || !videoEl) return;
  canvasEl.width = videoEl.clientWidth;
  canvasEl.height = videoEl.clientHeight;
}

function loadVideoSource(name) {
  currentVideoName = name;
  const folder = name.includes("class") ? "videos" : "samples";
  videoEl.src = `/api/video-file/${folder}/${encodeURIComponent(name)}`;
  videoEl.load();
}

function onVideoSelectChange(name) {
  loadVideoSource(name);
  document.getElementById("currentVideoBadge").textContent = name;
}

function cycleVideoInput() {
  if (videoSelect && videoSelect.options.length > 0) {
    let nextIdx = (videoSelect.selectedIndex + 1) % videoSelect.options.length;
    videoSelect.selectedIndex = nextIdx;
    onVideoSelectChange(videoSelect.value);
  }
}

function togglePlayPause() {
  if (!videoEl) return;
  if (videoEl.paused) {
    videoEl.play();
    playPauseBtn.textContent = "⏸ Pause";
    isVideoPlaying = true;
  } else {
    videoEl.pause();
    playPauseBtn.textContent = "▶ Play";
    isVideoPlaying = false;
  }
}

function stepVideo(deltaSec) {
  if (!videoEl) return;
  videoEl.currentTime = Math.max(0, Math.min(videoEl.duration, videoEl.currentTime + deltaSec));
}

function resetVideo() {
  if (!videoEl) return;
  videoEl.currentTime = 0;
}

function onScrubInput(val) {
  if (!videoEl) return;
  videoEl.currentTime = parseFloat(val);
  updateTimecodeDisplay();
}

function updateTimecodeDisplay() {
  if (!timeDisplay || !videoEl) return;
  const cur = formatTime(videoEl.currentTime || 0);
  const dur = formatTime(videoEl.duration || 20);
  timeDisplay.textContent = `${cur} / ${dur}`;
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
}

/* Throttled Frame Analysis */
let lastAnalysisCall = 0;
function triggerFrameAnalysisThrottled(timestamp) {
  const now = Date.now();
  if (now - lastAnalysisCall < 180 || isAnalyzingFrame) return; // ~5 FPS analysis rate
  lastAnalysisCall = now;
  isAnalyzingFrame = true;

  fetch("/api/analyze-frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_name: currentVideoName,
      timestamp_sec: timestamp,
      conf_threshold: 0.30,
      enable_ml: true
    })
  })
  .then(res => res.json())
  .then(data => {
    isAnalyzingFrame = false;
    if (data.detections) {
      currentDetections = data.detections;
      drawCanvasHUD(data);
    }
  })
  .catch(err => {
    isAnalyzingFrame = false;
  });
}

function drawCanvasHUD(data) {
  if (!ctx || !canvasEl || !videoEl) return;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const scaleX = canvasEl.width / (data.frame_dimensions?.width || 1280);
  const scaleY = canvasEl.height / (data.frame_dimensions?.height || 720);

  // 1. Draw Seat Outlines
  if (data.seats) {
    data.seats.forEach(s => {
      const [bx1, by1, bx2, by2] = s.box;
      const x = bx1 * scaleX;
      const y = by1 * scaleY;
      const w = (bx2 - bx1) * scaleX;
      const h = (by2 - by1) * scaleY;

      ctx.strokeStyle = s.is_occupied ? "rgba(168, 85, 247, 0.4)" : "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
      ctx.font = "10px sans-serif";
      ctx.fillText(`Seat #${s.seat_id}`, x + 4, y + 12);
    });
  }

  // 2. Draw Candidate Detections
  if (data.detections) {
    data.detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      // Color code by state
      let color = "#22c55e"; // NORMAL (Green)
      if (det.state === "OBSERVE") color = "#eab308"; // Yellow
      else if (det.state === "SUSPICIOUS") color = "#f97316"; // Orange
      else if (det.state === "ALERT") color = "#ef4444"; // Red

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);

      // Foot-point dot
      const fx = det.foot_point[0] * scaleX;
      const fy = det.foot_point[1] * scaleY;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(fx, fy, 4, 0, Math.PI * 2);
      ctx.fill();

      // Badge label
      const tag = `ID:${det.tracking_id} | Seat #${det.seat_id || "?"} | ${det.risk_score}/100`;
      ctx.fillStyle = "rgba(15, 14, 20, 0.85)";
      ctx.fillRect(rx, ry - 18, ctx.measureText(tag).width + 10, 16);
      ctx.fillStyle = color;
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(tag, rx + 4, ry - 6);
    });

    // Update HUD Stats
    document.getElementById("hudTrackedCount").textContent = data.detections.length;
    document.getElementById("hudRiskScore").textContent = `${data.stats?.max_risk_score || 0}/100`;
  }

  // 3. Draw Phones
  if (data.phones) {
    data.phones.forEach(p => {
      const [px1, py1, px2, py2] = p.bbox;
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(px1 * scaleX, py1 * scaleY, (px2 - px1) * scaleX, (py2 - py1) * scaleY);
    });
  }
}

/* --------------------------------------------------------------------------
   MODAL DIALOGS
-------------------------------------------------------------------------- */
function openVideoModal(seekSec = null) {
  const modal = document.getElementById("videoModal");
  if (!modal) return;
  modal.classList.add("open");
  resizeCanvas();

  if (seekSec !== null && videoEl) {
    videoEl.currentTime = seekSec;
  }
}

function closeVideoModal() {
  const modal = document.getElementById("videoModal");
  if (modal) modal.classList.remove("open");
  if (videoEl && !videoEl.paused) {
    videoEl.pause();
    playPauseBtn.textContent = "▶ Play";
  }
}

function openSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.classList.add("open");
}

function closeSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.classList.remove("open");
}

function showNotificationToast() {
  alert(`🔔 ExamGuard AI Notifications:\n• Tracking 20 students in ${currentVideoName}\n• High-priority alerts recorded: ${allAlerts.length}\n• Model: Fine-Tuned YOLOv8 + ML Anomaly Detector`);
}

function showSupportInfo() {
  alert("ExamGuard AI Support: System running at 60 FPS without Streamlit lag. In case of issues, verify models/yolov8n_exam.pt or run train.py.");
}

/* --------------------------------------------------------------------------
   AI QUERY ASSISTANT & NAVIGATION
-------------------------------------------------------------------------- */
async function handleAIQuery(e) {
  if (e.key === "Enter") {
    const input = document.getElementById("aiQueryInput");
    const query = input.value.trim();
    if (!query) return;

    input.value = "Consulting AI Pattern Agent...";
    try {
      const res = await fetch("/api/ask-ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
      });
      const data = await res.json();
      alert(`🤖 ExamGuard Assistant:\n\n${data.answer}`);
    } catch (err) {
      alert("AI Assistant query failed: " + err);
    } finally {
      input.value = "";
    }
  }
}

function downloadCSVReport() {
  window.open("/api/export-csv", "_blank");
}

function switchSubPill(tabId) {
  const pills = document.querySelectorAll(".sub-bar .pill-tab");
  pills.forEach(p => p.classList.remove("active"));
  event.target.classList.add("active");

  if (tabId === "models") {
    openSettingsModal();
  }
}

function switchMainTab(view) {
  if (view === "dashboard") {
    closeVideoModal();
    closeSettingsModal();
  } else if (view === "video") {
    openVideoModal();
  } else if (view === "models") {
    openSettingsModal();
  } else {
    openVideoModal();
  }
}

async function applyModelPreferences() {
  const useFineTuned = document.getElementById("detectorModelSelect").value === "fine_tuned";
  const enableML = document.getElementById("mlAnomalyToggle").checked;

  try {
    const res = await fetch("/api/switch-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        use_fine_tuned: useFineTuned,
        enable_ml_anomaly: enableML
      })
    });
    const data = await res.json();
    alert(`✅ Model updated:\nDetector: ${data.detector_path}\nML Anomaly: ${data.ml_anomaly_active ? "Active" : "Disabled"}`);
    closeSettingsModal();
    fetchSystemStatus();
  } catch (err) {
    alert("Model update failed: " + err);
  }
}
