/**
 * GLDv2 Visual Landmark Search Engine - Frontend Client
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabImageBtn = document.getElementById("tabImageBtn");
  const tabTextBtn = document.getElementById("tabTextBtn");
  const imageTab = document.getElementById("imageTab");
  const textTab = document.getElementById("textTab");

  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const imagePreview = document.getElementById("imagePreview");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const removeImageBtn = document.getElementById("removeImageBtn");

  const textQueryInput = document.getElementById("textQueryInput");
  const clearTextBtn = document.getElementById("clearTextBtn");
  const chipButtons = document.querySelectorAll(".chip-btn");

  const topKSelect = document.getElementById("topKSelect");
  const indexTypeSelect = document.getElementById("indexTypeSelect");
  const tuningParamSelect = document.getElementById("tuningParamSelect");
  const tuningParamLabel = document.getElementById("tuningParamLabel");
  const searchSubmitBtn = document.getElementById("searchSubmitBtn");
  const btnSpinner = searchSubmitBtn.querySelector(".btn-spinner");
  const btnText = searchSubmitBtn.querySelector(".btn-text");

  const errorBanner = document.getElementById("errorBanner");
  const errorMessage = document.getElementById("errorMessage");
  const closeErrorBtn = document.getElementById("closeErrorBtn");

  const resultsSection = document.getElementById("resultsSection");
  const queryDesc = document.getElementById("queryDesc");
  const metricEmbTime = document.getElementById("metricEmbTime");
  const metricSearchTime = document.getElementById("metricSearchTime");
  const metricTotalTime = document.getElementById("metricTotalTime");

  const topCandidateCard = document.getElementById("topCandidateCard");
  const candName = document.getElementById("candName");
  const candId = document.getElementById("candId");
  const candSplit = document.getElementById("candSplit");
  const candScore = document.getElementById("candScore");
  const candMargin = document.getElementById("candMargin");
  const candMatches = document.getElementById("candMatches");
  const candHierarchyWrap = document.getElementById("candHierarchyWrap");
  const candHierarchy = document.getElementById("candHierarchy");
  const candWikiBtn = document.getElementById("candWikiBtn");

  const matchesCount = document.getElementById("matchesCount");
  const matchesGrid = document.getElementById("matchesGrid");

  const pillVectors = document.getElementById("pillVectors");
  const pillDevice = document.getElementById("pillDevice");

  // State
  let activeTab = "image"; // 'image' | 'text'
  let selectedFile = null;

  // 1. Fetch system info on startup
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/info");
      if (res.ok) {
        const data = await res.json();
        if (data.total_vectors) {
          pillVectors.textContent = `${data.total_vectors.toLocaleString()} Vectors`;
        }
        if (data.device) {
          pillDevice.innerHTML = `Device: <strong>${data.device.toUpperCase()}</strong>`;
        }
        if (data.available_indices && indexTypeSelect) {
          const ivfCount = data.ivf_vectors ? ` (${(data.ivf_vectors / 1e6).toFixed(2)}M)` : "";
          const hnswCount = data.hnsw_vectors ? ` (${(data.hnsw_vectors / 1e3).toFixed(0)}k)` : "";
          const currentVal = indexTypeSelect.value;
          indexTypeSelect.innerHTML = "";
          if (data.available_indices.includes("ivfpq")) {
            const opt = document.createElement("option");
            opt.value = "ivfpq";
            opt.textContent = `IVF-PQ Inverted Index${ivfCount}`;
            if (currentVal === "ivfpq") opt.selected = true;
            indexTypeSelect.appendChild(opt);
          }
          if (data.available_indices.includes("hnswpq")) {
            const opt = document.createElement("option");
            opt.value = "hnswpq";
            opt.textContent = `HNSW-PQ Graph Index${hnswCount}`;
            if (currentVal === "hnswpq") opt.selected = true;
            indexTypeSelect.appendChild(opt);
          }
        }
      }
    } catch (e) {
      console.warn("Could not fetch system info:", e);
    }
  }
  fetchSystemInfo();

  // Algorithm selection change handler
  if (indexTypeSelect) {
    indexTypeSelect.addEventListener("change", () => {
      const isHnsw = indexTypeSelect.value === "hnswpq";
      if (tuningParamLabel) {
        tuningParamLabel.textContent = isHnsw ? "Search Depth (efSearch):" : "Voronoi Probes (nprobe):";
      }
      if (tuningParamSelect) {
        tuningParamSelect.innerHTML = isHnsw
          ? `<option value="32">32 (Fastest)</option>
             <option value="64" selected>64 (Balanced)</option>
             <option value="128">128 (High Recall)</option>
             <option value="256">256 (Exhaustive)</option>`
          : `<option value="16">16 (Fastest)</option>
             <option value="32">32 (Balanced)</option>
             <option value="64" selected>64 (High Recall)</option>
             <option value="128">128 (Max Precision)</option>`;
      }
    });
  }

  // 2. Tab Navigation
  function switchTab(mode) {
    activeTab = mode;
    if (mode === "image") {
      tabImageBtn.classList.add("active");
      tabTextBtn.classList.remove("active");
      imageTab.classList.add("active");
      textTab.classList.remove("active");
    } else {
      tabTextBtn.classList.add("active");
      tabImageBtn.classList.remove("active");
      textTab.classList.add("active");
      imageTab.classList.remove("active");
      textQueryInput.focus();
    }
  }

  tabImageBtn.addEventListener("click", () => switchTab("image"));
  tabTextBtn.addEventListener("click", () => switchTab("text"));

  // 3. File Selection & Drag-and-Drop
  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      showError("Please select a valid image file (JPEG, PNG, WEBP).");
      return;
    }
    selectedFile = file;
    previewName.textContent = file.name;
    previewSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzoneEmpty.style.display = "none";
      dropzonePreview.style.display = "flex";
      hideError();
    };
    reader.readAsDataURL(file);
  }

  function clearSelectedImage() {
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    dropzonePreview.style.display = "none";
    dropzoneEmpty.style.display = "block";
  }

  removeImageBtn.addEventListener("click", clearSelectedImage);

  // Drag and drop handlers
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Paste image from clipboard
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf("image") === 0) {
        const blob = item.getAsFile();
        switchTab("image");
        handleFileSelected(blob);
        break;
      }
    }
  });

  // 4. Text Input & Sample Chips
  textQueryInput.addEventListener("input", () => {
    clearTextBtn.style.display = textQueryInput.value ? "block" : "none";
  });

  clearTextBtn.addEventListener("click", () => {
    textQueryInput.value = "";
    clearTextBtn.style.display = "none";
    textQueryInput.focus();
  });

  textQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      triggerSearch();
    }
  });

  chipButtons.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      switchTab("text");
      textQueryInput.value = query;
      clearTextBtn.style.display = "block";
      triggerSearch();
    });
  });

  // 5. Error Banner Controls
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.style.display = "flex";
  }

  function hideError() {
    errorBanner.style.display = "none";
  }

  closeErrorBtn.addEventListener("click", hideError);

  // 6. Search Execution
  searchSubmitBtn.addEventListener("click", triggerSearch);

  function setLoading(loading) {
    if (loading) {
      searchSubmitBtn.disabled = true;
      btnSpinner.style.display = "inline-block";
      btnText.textContent = "Searching...";
      hideError();
    } else {
      searchSubmitBtn.disabled = false;
      btnSpinner.style.display = "none";
      btnText.textContent = "Search Landmark";
    }
  }

  async function triggerSearch() {
    const topK = parseInt(topKSelect.value, 10);
    const indexType = indexTypeSelect ? indexTypeSelect.value : "ivfpq";
    const tuningVal = parseInt(tuningParamSelect ? tuningParamSelect.value : 64, 10);
    const nprobe = indexType === "ivfpq" ? tuningVal : 64;
    const efSearch = indexType === "hnswpq" ? tuningVal : 64;

    if (activeTab === "image") {
      if (!selectedFile) {
        showError("Please upload or drag an image first.");
        return;
      }
      await executeImageSearch(selectedFile, topK, nprobe, indexType, efSearch);
    } else {
      const text = textQueryInput.value.trim();
      if (!text) {
        showError("Please enter a landmark text description.");
        return;
      }
      await executeTextSearch(text, topK, nprobe, indexType, efSearch);
    }
  }

  async function executeTextSearch(text, topK, nprobe, indexType, efSearch) {
    setLoading(true);
    try {
      const response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          top_k: topK,
          nprobe: nprobe,
          index_type: indexType,
          ef_search: efSearch,
          rerank: false,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Search request failed");
      }

      const data = await response.json();
      renderResults(data);
    } catch (e) {
      showError(e.message || "Failed to execute text search.");
    } finally {
      setLoading(false);
    }
  }

  async function executeImageSearch(file, topK, nprobe, indexType, efSearch) {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("top_k", topK);
      formData.append("nprobe", nprobe);
      formData.append("index_type", indexType);
      formData.append("ef_search", efSearch);
      const isGeom = document.getElementById("geomCheckbox") ? document.getElementById("geomCheckbox").checked : false;
      formData.append("rerank", "false");
      formData.append("rerank_geometric", isGeom ? "true" : "false");

      const response = await fetch("/api/search/image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Search request failed");
      }

      const data = await response.json();
      renderResults(data);
    } catch (e) {
      showError(e.message || "Failed to execute image search.");
    } finally {
      setLoading(false);
    }
  }

  // 7. Render Search Results
  function renderResults(data) {
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    // Metrics
    const algoBadge = data.index_type === "hnswpq"
      ? ` [HNSW-PQ, efSearch=${data.ef_search || 64}]`
      : ` [IVF-PQ, nprobe=${data.nprobe || 64}]`;
    queryDesc.textContent = (data.query_desc || "Query") + algoBadge;
    metricEmbTime.textContent = `${data.embedding_latency_ms || 0} ms`;
    metricSearchTime.textContent = `${data.search_latency_ms || 0} ms`;
    metricTotalTime.textContent = `${data.total_latency_ms || 0} ms`;

    // Top Candidate
    if (data.voting_summary && data.voting_summary.landmark_name) {
      const v = data.voting_summary;
      topCandidateCard.style.display = "block";
      candName.textContent = v.landmark_name;
      candId.textContent = v.landmark_id;
      candSplit.textContent = v.split || "train";
      candScore.textContent = v.total_vote_score;
      candMargin.textContent = `Confidence: +${v.confidence_margin} Margin`;
      candMatches.textContent = `${v.match_count} image matches`;

      if (v.hierarchical_label && v.hierarchical_label !== "nan") {
        candHierarchyWrap.style.display = "inline";
        candHierarchy.textContent = v.hierarchical_label;
      } else {
        candHierarchyWrap.style.display = "none";
      }

      if (v.category_url) {
        candWikiBtn.href = v.category_url;
        candWikiBtn.style.display = "inline-flex";
      } else {
        candWikiBtn.style.display = "none";
      }
    } else {
      topCandidateCard.style.display = "none";
    }

    // Nearest Neighbor Cards Grid
    matchesGrid.innerHTML = "";
    const matches = data.matches || [];
    matchesCount.textContent = `${matches.length} matches retrieved (top_k=${data.top_k}, nprobe=${data.nprobe})`;

    matches.forEach((m) => {
      const card = document.createElement("div");
      card.className = "match-card";

      // Normalize score for progress bar (cosine similarity ~0.20 to ~0.70 mapped to 20%-100%)
      const pct = Math.min(100, Math.max(10, Math.round(m.score * 150)));

      const isVerified = Boolean(m.geometric_verified);
      const scoreVal = m.final_score !== undefined ? m.final_score : m.score;

      card.innerHTML = `
        <div class="match-top">
          <div style="display:flex; align-items:center; gap:6px;">
            <span class="rank-badge ${m.rank <= 3 ? "rank-top" : ""}">#${m.rank}</span>
            ${isVerified ? `<span style="background:#10b981; color:#fff; font-size:10px; font-weight:700; padding:2px 6px; border-radius:4px; letter-spacing:0.5px;">✓ GEOM MATCH</span>` : ""}
          </div>
          <span class="score-display">${isVerified ? "Score" : "Sim"}: ${scoreVal.toFixed(4)}</span>
        </div>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width: ${pct}%"></div>
        </div>
        <div class="match-name">${escapeHtml(m.landmark_name)}</div>
        <div class="match-details">
          <span>Landmark ID: <strong>${m.landmark_id}</strong></span>
          <span>Image ID: <strong>${m.image_id}</strong></span>
          <span>Dataset Split: <strong>${m.split}</strong></span>
          ${isVerified ? `<span>Inliers: <strong>${m.inliers}</strong> (${Math.round((m.inlier_ratio || 0) * 100)}% ratio)</span>` : ""}
          ${m.hierarchical_label && m.hierarchical_label !== "nan" ? `<span>Category: <em>${escapeHtml(m.hierarchical_label)}</em></span>` : ""}
        </div>
        ${
          m.category_url
            ? `<div class="match-footer">
                 <a href="${m.category_url}" target="_blank" rel="noopener noreferrer" class="match-link">Wikimedia Category ↗</a>
               </div>`
            : ""
        }
      `;
      matchesGrid.appendChild(card);
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
