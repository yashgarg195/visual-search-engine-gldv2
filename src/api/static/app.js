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

  // AI Location Intelligence & Verification Elements
  const locationIntelCard = document.getElementById("locationIntelCard");
  const locGeoTag = document.getElementById("locGeoTag");
  const locProminentName = document.getElementById("locProminentName");
  const locSpecificWrap = document.getElementById("locSpecificWrap");
  const locSpecificName = document.getElementById("locSpecificName");
  const locSummary = document.getElementById("locSummary");
  const refGalleryGrid = document.getElementById("refGalleryGrid");
  const galleryScrollPrev = document.getElementById("galleryScrollPrev");
  const galleryScrollNext = document.getElementById("galleryScrollNext");

  const coordsDisplay = document.getElementById("coordsDisplay");


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
          const hnswCount = data.hnsw_vectors ? ` (${(data.hnsw_vectors / 1e6).toFixed(2)}M)` : "";
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
          if (data.available_indices.includes("ensemble")) {
            const opt = document.createElement("option");
            opt.value = "ensemble";
            opt.textContent = `Ensemble (Hybrid: IVF + HNSW)`;
            if (currentVal === "ensemble") opt.selected = true;
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
      const val = indexTypeSelect.value;
      const isHnsw = val === "hnswpq";
      const isEnsemble = val === "ensemble";
      if (tuningParamLabel) {
        if (isEnsemble) {
          tuningParamLabel.textContent = "Probes / Depth:";
        } else if (isHnsw) {
          tuningParamLabel.textContent = "Search Depth (efSearch):";
        } else {
          tuningParamLabel.textContent = "Voronoi Probes (nprobe):";
        }
      }
      if (tuningParamSelect) {
        if (isEnsemble) {
          tuningParamSelect.innerHTML = `
            <option value="32">32 (Fast Hybrid)</option>
            <option value="64" selected>64 (Balanced Hybrid)</option>
            <option value="128">128 (Max Recall Hybrid)</option>`;
        } else if (isHnsw) {
          tuningParamSelect.innerHTML = `
            <option value="32">32 (Fastest)</option>
            <option value="64" selected>64 (Balanced)</option>
            <option value="128">128 (High Recall)</option>
            <option value="256">256 (Exhaustive)</option>`;
        } else {
          tuningParamSelect.innerHTML = `
            <option value="16">16 (Fastest)</option>
            <option value="32">32 (Balanced)</option>
            <option value="64" selected>64 (High Recall)</option>
            <option value="128">128 (Max Precision)</option>`;
        }
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
      // Auto-trigger visual landmark search
      triggerSearch();
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

  function scrollReferenceGallery(direction) {
    if (!refGalleryGrid) return;
    refGalleryGrid.scrollBy({ left: direction * Math.max(220, refGalleryGrid.clientWidth * 0.72), behavior: "smooth" });
  }

  galleryScrollPrev?.addEventListener("click", () => scrollReferenceGallery(-1));
  galleryScrollNext?.addEventListener("click", () => scrollReferenceGallery(1));

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

  const metricFusionWrap = document.getElementById("metricFusionWrap");
  const metricFusionTime = document.getElementById("metricFusionTime");

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
          rerank: true,
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
      formData.append("rerank", "true");

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
    let algoBadge = "";
    if (data.index_type === "ensemble") {
      algoBadge = ` [Ensemble: IVF + HNSW, RRF]`;
    } else if (data.index_type === "hnswpq") {
      algoBadge = ` [HNSW-PQ, efSearch=${data.ef_search || 64}]`;
    } else {
      algoBadge = ` [IVF-PQ, nprobe=${data.nprobe || 64}]`;
    }
    queryDesc.textContent = (data.query_desc || "Query") + algoBadge;
    metricEmbTime.textContent = `${data.embedding_latency_ms || 0} ms`;
    metricSearchTime.textContent = `${data.search_latency_ms || 0} ms`;

    if (metricFusionWrap && metricFusionTime) {
      if (data.fusion_latency_ms !== undefined && data.fusion_latency_ms >= 0) {
        metricFusionWrap.style.display = "inline-flex";
        metricFusionTime.textContent = `${data.fusion_latency_ms} ms`;
      } else {
        metricFusionWrap.style.display = "none";
      }
    }

    metricTotalTime.textContent = `${data.total_latency_ms || 0} ms`;

    // AI Location Intelligence (Gemini 3.8 Flash + Geocoding)
    if (data.location_info && (data.location_info.prominent_attraction || data.location_info.city || data.location_info.country)) {
      const loc = data.location_info;
      locationIntelCard.style.display = "block";
      locProminentName.textContent = loc.prominent_attraction || "Identified Landmark";

      const geoParts = [loc.city, loc.country].filter(p => p && p !== "nan");
      locGeoTag.textContent = geoParts.length > 0 ? `📍 ${geoParts.join(", ")}` : "📍 World Landmark";

      if (loc.specific_subject && loc.specific_subject.toLowerCase() !== (loc.prominent_attraction || "").toLowerCase()) {
        locSpecificWrap.style.display = "flex";
        locSpecificName.textContent = loc.specific_subject;
      } else {
        locSpecificWrap.style.display = "none";
      }

      locSummary.textContent = loc.summary || "";

      // Google Maps Pin embedded directly on page
      const gmapsIframe = document.getElementById("gmapsIframe");
      const lat = parseFloat(loc.latitude);
      const lon = parseFloat(loc.longitude);
      let mapQuery = "";

      if (!isNaN(lat) && !isNaN(lon) && (lat !== 0 || lon !== 0)) {
        const latitudeHemisphere = lat >= 0 ? "N" : "S";
        const longitudeHemisphere = lon >= 0 ? "E" : "W";
        coordsDisplay.textContent = `${Math.abs(lat).toFixed(4)}° ${latitudeHemisphere}, ${Math.abs(lon).toFixed(4)}° ${longitudeHemisphere}`;
        mapQuery = `${lat},${lon}`;
      } else if (loc.prominent_attraction) {
        coordsDisplay.textContent = "Coordinates estimated";
        mapQuery = encodeURIComponent(`${loc.prominent_attraction} ${loc.city || ""}`.trim());
      }

      if (gmapsIframe && mapQuery) {
        gmapsIframe.src = `https://maps.google.com/maps?q=${mapQuery}&t=m&z=15&output=embed&iwloc=near`;
      }

      // Canonical Reference Gallery (Horizontally Scrollable Card)
      refGalleryGrid.innerHTML = "";
      const refs = data.reference_images || [];
      if (refs.length === 0) {
        refGalleryGrid.innerHTML = `<span style="font-size:0.8rem; color:#94a3b8; padding: 1rem 0;">No canonical photos found.</span>`;
      } else {
        refs.forEach((ref) => {
          const refCard = document.createElement("div");
          refCard.className = "ref-card";
          refCard.innerHTML = `
            <div class="ref-img-wrap">
              <img src="${ref.thumb_url || ref.original_url}" alt="${escapeHtml(ref.title)}" loading="lazy" />
              <span class="ref-badge canonical">Canonical Photo</span>
            </div>
            <div class="ref-info">
              <span class="ref-title" title="${escapeHtml(ref.title)}">${escapeHtml(ref.title)}</span>
            </div>
          `;
          refGalleryGrid.appendChild(refCard);
        });
      }
    } else {
      locationIntelCard.style.display = "none";
    }

    // Top Candidate
    if (data.voting_summary && data.voting_summary.landmark_name) {
      const v = data.voting_summary;
      topCandidateCard.style.display = "block";
      candName.textContent = v.landmark_name;
      candId.textContent = v.landmark_id;
      candSplit.textContent = v.split || "train";
      candScore.textContent = v.total_vote_score;
      candMargin.textContent = `Lead: +${v.confidence_margin} over next candidate`;
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

      const scoreVal = m.final_score !== undefined ? m.final_score : m.score;
      // Normalize score for progress bar (0.20 to 0.70 mapped to 20%-100%)
      const pct = Math.min(100, Math.max(10, Math.round(scoreVal * 150)));

      const isVicinityVerified = Boolean(m.is_vicinity_verified);
      const isConsensus = Boolean(m.consensus);

      // Build thumbnail URL from image_id and split
      const imgId = m.image_id || "";
      const imgSplit = m.split || "train";
      const thumbUrl = (imgId.length >= 3) ? `/api/image/${imgSplit}/${imgId[0]}/${imgId[1]}/${imgId[2]}/${imgId}.jpg` : "";

      card.innerHTML = `
        <div class="match-card-inner">
          ${thumbUrl ? `<div class="match-thumb-wrap">
            <img class="match-thumb" src="${thumbUrl}" alt="${escapeHtml(m.landmark_name)}" loading="lazy"
                 onerror="this.parentElement.style.display='none'" />
          </div>` : ""}
          <div class="match-content">
            <div class="match-top">
              <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                <span class="rank-badge ${m.rank <= 3 ? "rank-top" : ""}">#${m.rank}</span>
                ${isVicinityVerified ? `<span class="vicinity-badge">📍 Coordinate-supported</span>` : ""}
                ${isConsensus ? `<span class="consensus-badge">⭐ CONSENSUS</span>` : ""}
              </div>
              <span class="score-display">Evidence score ${scoreVal.toFixed(4)}</span>
            </div>
            <div class="score-bar-bg">
              <div class="score-bar-fill" style="width: ${pct}%"></div>
            </div>
            <div class="match-name">${escapeHtml(m.landmark_name)}</div>
            <div class="match-details">
              <span>Landmark ID: <strong>${m.landmark_id}</strong></span>
              <span>Image ID: <strong>${m.image_id}</strong></span>
              <span>Dataset Split: <strong>${m.split}</strong></span>
              ${m.vicinity_cluster ? `<span>Vicinity: <strong>${escapeHtml(m.vicinity_cluster)}</strong></span>` : ""}
              ${m.distance_km !== null && m.distance_km !== undefined ? `<span>Distance: <strong>${m.distance_km} km away</strong></span>` : ""}
              ${m.visual_sim !== undefined ? `<span>Breakdown: <strong>Vis ${Math.round(m.visual_sim * 100)}% | Vic ${Math.round(m.vicinity_score * 100)}% | Geo ${Math.round(m.spatial_score * 100)}%</strong></span>` : ""}
              ${m.sources && m.sources.length > 0 ? `<span>Retrieved By: <strong>${m.sources.join(' + ').toUpperCase()}</strong></span>` : ""}
              ${m.hierarchical_label && m.hierarchical_label !== "nan" ? `<span>Category: <em>${escapeHtml(m.hierarchical_label)}</em></span>` : ""}
            </div>

            ${
              m.category_url
                ? `<div class="match-footer">
                     <a href="${m.category_url}" target="_blank" rel="noopener noreferrer" class="match-link">Wikimedia Category ↗</a>
                   </div>`
                : ""
            }
          </div>
        </div>
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
