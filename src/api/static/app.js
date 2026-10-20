/**
 * GLDv2 Visual Landmark Search Engine - Frontend Client
 */

document.addEventListener("DOMContentLoaded", () => {
  const ICONS = {
    pin: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 21s7-5.2 7-10.5A7 7 0 0 0 5 10.5C5 15.8 12 21 12 21z"/><circle cx="12" cy="10.5" r="2.5"/></svg>`,
    star: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="m12 3.6 2.5 5.1 5.6.8-4 4 .9 5.6-5-2.7-5 2.7.9-5.6-4-4 5.6-.8z"/></svg>`,
  };

  const MATCHES_PER_PAGE = 6;

  // ---------- Theme ----------
  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (themeToggle) {
      const isDark = theme === "dark";
      themeToggle.setAttribute("aria-pressed", String(isDark));
      themeToggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    }
  }

  applyTheme(root.getAttribute("data-theme") === "dark" ? "dark" : "light");

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        localStorage.setItem("ll-theme", next);
      } catch (e) {
        /* storage unavailable — theme still applies for this session */
      }
    });
  }

  // ---------- Elements ----------
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
  const optionsSummary = document.getElementById("optionsSummary");
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
  const metricFusionWrap = document.getElementById("metricFusionWrap");
  const metricFusionTime = document.getElementById("metricFusionTime");

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
  const matchesPagination = document.getElementById("matchesPagination");
  const pagePrev = document.getElementById("pagePrev");
  const pageNext = document.getElementById("pageNext");
  const pageList = document.getElementById("pageList");
  const pageStatus = document.getElementById("pageStatus");
  const srStatus = document.getElementById("srStatus");

  function announce(msg) {
    if (srStatus) srStatus.textContent = msg;
  }

  const pillVectors = document.getElementById("pillVectors");
  const pillDevice = document.getElementById("pillDevice");

  const locationIntelCard = document.getElementById("locationIntelCard");
  const locGeoText = document.getElementById("locGeoText");
  const locProminentName = document.getElementById("locProminentName");
  const locSpecificWrap = document.getElementById("locSpecificWrap");
  const locSpecificName = document.getElementById("locSpecificName");
  const locSummary = document.getElementById("locSummary");
  const refGalleryGrid = document.getElementById("refGalleryGrid");
  const galleryScrollPrev = document.getElementById("galleryScrollPrev");
  const galleryScrollNext = document.getElementById("galleryScrollNext");
  const coordsDisplay = document.getElementById("coordsDisplay");

  const sysBtn = document.getElementById("sysBtn");
  const sysDrawer = document.getElementById("sysDrawer");
  const sysBackdrop = document.getElementById("sysBackdrop");
  const sysClose = document.getElementById("sysClose");

  // ---------- State ----------
  let activeTab = "image"; // 'image' | 'text'
  let selectedFile = null;
  let currentMatches = [];
  let currentPage = 1;
  let lastQueryMeta = {};

  // ---------- System drawer ----------
  function openDrawer() {
    sysDrawer.hidden = false;
    sysBackdrop.hidden = false;
    sysBtn.setAttribute("aria-expanded", "true");
    sysClose.focus();
    document.addEventListener("keydown", onDrawerKeydown);
  }

  function closeDrawer() {
    sysDrawer.hidden = true;
    sysBackdrop.hidden = true;
    sysBtn.setAttribute("aria-expanded", "false");
    document.removeEventListener("keydown", onDrawerKeydown);
    sysBtn.focus();
  }

  function onDrawerKeydown(e) {
    if (e.key === "Escape") {
      closeDrawer();
      return;
    }
    // Keep Tab inside the drawer while it behaves as a modal dialog.
    if (e.key !== "Tab") return;
    const focusables = sysDrawer.querySelectorAll(
      'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  sysBtn.addEventListener("click", () => {
    if (sysDrawer.hidden) openDrawer();
    else closeDrawer();
  });
  sysClose.addEventListener("click", closeDrawer);
  sysBackdrop.addEventListener("click", closeDrawer);

  // ---------- System info ----------
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) return;
      const data = await res.json();

      if (data.total_vectors) {
        pillVectors.textContent = data.total_vectors.toLocaleString();
      }
      if (data.device) {
        pillDevice.textContent = String(data.device).toUpperCase();
      }
      if (data.available_indices && indexTypeSelect) {
        const ivfCount = data.ivf_vectors ? ` (${(data.ivf_vectors / 1e6).toFixed(2)}M)` : "";
        const hnswCount = data.hnsw_vectors ? ` (${(data.hnsw_vectors / 1e6).toFixed(2)}M)` : "";
        const currentVal = indexTypeSelect.value;
        indexTypeSelect.innerHTML = "";

        const addOption = (value, label) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = label;
          if (currentVal === value) opt.selected = true;
          indexTypeSelect.appendChild(opt);
        };

        if (data.available_indices.includes("ivfpq")) addOption("ivfpq", `IVF-PQ Inverted Index${ivfCount}`);
        if (data.available_indices.includes("hnswpq")) addOption("hnswpq", `HNSW-PQ Graph Index${hnswCount}`);
        if (data.available_indices.includes("ensemble")) addOption("ensemble", "Ensemble (IVF + HNSW)");
        updateOptionsSummary();
      }
    } catch (e) {
      console.warn("Could not fetch system info:", e);
    }
  }
  fetchSystemInfo();

  // ---------- Options summary (collapsed state still shows the choice) ----------
  function updateOptionsSummary() {
    if (!optionsSummary) return;
    const algoText = {
      ivfpq: "IVF-PQ",
      hnswpq: "HNSW-PQ",
      ensemble: "Ensemble",
    }[indexTypeSelect ? indexTypeSelect.value : "ivfpq"] || "IVF-PQ";
    const k = topKSelect ? topKSelect.value : "10";
    optionsSummary.textContent = `${algoText} · ${k} results`;
  }

  if (indexTypeSelect) {
    indexTypeSelect.addEventListener("change", () => {
      const val = indexTypeSelect.value;
      const isHnsw = val === "hnswpq";
      const isEnsemble = val === "ensemble";

      if (tuningParamLabel) {
        if (isEnsemble) tuningParamLabel.textContent = "Probes / depth";
        else if (isHnsw) tuningParamLabel.textContent = "Search depth (efSearch)";
        else tuningParamLabel.textContent = "Voronoi probes (nprobe)";
      }

      if (tuningParamSelect) {
        if (isEnsemble) {
          tuningParamSelect.innerHTML = `
            <option value="32">32 (Fast hybrid)</option>
            <option value="64" selected>64 (Balanced hybrid)</option>
            <option value="128">128 (Max recall hybrid)</option>`;
        } else if (isHnsw) {
          tuningParamSelect.innerHTML = `
            <option value="32">32 (Fastest)</option>
            <option value="64" selected>64 (Balanced)</option>
            <option value="128">128 (High recall)</option>
            <option value="256">256 (Exhaustive)</option>`;
        } else {
          tuningParamSelect.innerHTML = `
            <option value="16">16 (Fastest)</option>
            <option value="32">32 (Balanced)</option>
            <option value="64" selected>64 (High recall)</option>
            <option value="128">128 (Max precision)</option>`;
        }
      }
      updateOptionsSummary();
    });
  }
  if (topKSelect) topKSelect.addEventListener("change", updateOptionsSummary);
  updateOptionsSummary();

  // ---------- Tabs ----------
  const tabButtons = [tabImageBtn, tabTextBtn];

  function switchTab(mode) {
    activeTab = mode;
    const isImage = mode === "image";

    tabImageBtn.classList.toggle("active", isImage);
    tabTextBtn.classList.toggle("active", !isImage);
    tabImageBtn.setAttribute("aria-selected", String(isImage));
    tabTextBtn.setAttribute("aria-selected", String(!isImage));
    // Roving tabindex: only the selected tab is in the tab order.
    tabImageBtn.setAttribute("tabindex", isImage ? "0" : "-1");
    tabTextBtn.setAttribute("tabindex", isImage ? "-1" : "0");
    imageTab.hidden = !isImage;
    textTab.hidden = isImage;
    imageTab.classList.toggle("active", isImage);
    textTab.classList.toggle("active", !isImage);
  }

  tabImageBtn.addEventListener("click", () => switchTab("image"));
  tabTextBtn.addEventListener("click", () => switchTab("text"));

  // Arrow/Home/End move between tabs and keep focus on the tab itself.
  tabButtons.forEach((tabBtn) => {
    tabBtn.addEventListener("keydown", (e) => {
      const idx = tabButtons.indexOf(e.currentTarget);
      let nextIdx = null;
      if (e.key === "ArrowRight") nextIdx = (idx + 1) % tabButtons.length;
      else if (e.key === "ArrowLeft") nextIdx = (idx - 1 + tabButtons.length) % tabButtons.length;
      else if (e.key === "Home") nextIdx = 0;
      else if (e.key === "End") nextIdx = tabButtons.length - 1;
      if (nextIdx === null) return;
      e.preventDefault();
      switchTab(nextIdx === 0 ? "image" : "text");
      tabButtons[nextIdx].focus();
    });
  });

  // ---------- File selection ----------
  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) handleFileSelected(e.target.files[0]);
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
      dropzoneEmpty.hidden = true;
      dropzonePreview.hidden = false;
      hideError();
    };
    reader.readAsDataURL(file);
  }

  function clearSelectedImage() {
    selectedFile = null;
    fileInput.value = "";
    imagePreview.removeAttribute("src");
    dropzonePreview.hidden = true;
    dropzoneEmpty.hidden = false;
  }

  removeImageBtn.addEventListener("click", clearSelectedImage);

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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileSelected(e.dataTransfer.files[0]);
  });

  window.addEventListener("paste", (e) => {
    const clip = e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData);
    if (!clip) return;
    for (const item of clip.items) {
      if (item.type.indexOf("image") === 0) {
        const blob = item.getAsFile();
        if (blob) {
          switchTab("image");
          handleFileSelected(blob);
        }
        break;
      }
    }
  });

  // ---------- Text input ----------
  textQueryInput.addEventListener("input", () => {
    clearTextBtn.hidden = !textQueryInput.value;
  });

  clearTextBtn.addEventListener("click", () => {
    textQueryInput.value = "";
    clearTextBtn.hidden = true;
    textQueryInput.focus();
  });

  textQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") triggerSearch();
  });

  chipButtons.forEach((chip) => {
    chip.addEventListener("click", () => {
      switchTab("text");
      textQueryInput.value = chip.getAttribute("data-query");
      clearTextBtn.hidden = false;
      triggerSearch();
    });
  });

  // ---------- Error banner ----------
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.hidden = false;
  }
  function hideError() {
    errorBanner.hidden = true;
  }
  closeErrorBtn.addEventListener("click", hideError);

  // ---------- Reference gallery scrolling ----------
  function scrollReferenceGallery(direction) {
    if (!refGalleryGrid) return;
    refGalleryGrid.scrollBy({
      left: direction * Math.max(220, refGalleryGrid.clientWidth * 0.72),
      behavior: "smooth",
    });
  }
  galleryScrollPrev?.addEventListener("click", () => scrollReferenceGallery(-1));
  galleryScrollNext?.addEventListener("click", () => scrollReferenceGallery(1));

  // ---------- Search ----------
  searchSubmitBtn.addEventListener("click", triggerSearch);

  function setLoading(loading) {
    searchSubmitBtn.disabled = loading;
    btnSpinner.hidden = !loading;
    btnText.textContent = loading ? "Searching…" : "Search";
    if (loading) hideError();
  }

  // Only the newest search may render. Older in-flight requests are aborted,
  // and any late response is dropped by the sequence guard.
  let inFlightController = null;
  let requestSeq = 0;

  async function parseErrorResponse(response) {
    const fallback = `Search failed (HTTP ${response.status})`;
    try {
      const err = await response.json();
      const detail = err && err.detail;
      if (typeof detail === "string" && detail) return detail;
      // FastAPI validation errors return detail as an array of objects.
      if (Array.isArray(detail)) {
        const msg = detail.map((d) => (d && d.msg) || "").filter(Boolean).join("; ");
        if (msg) return msg;
      }
      return fallback;
    } catch (_) {
      return fallback;
    }
  }

  async function triggerSearch() {
    const topK = parseInt(topKSelect.value, 10);
    const indexType = indexTypeSelect ? indexTypeSelect.value : "ivfpq";
    const tuningVal = parseInt(tuningParamSelect ? tuningParamSelect.value : 64, 10);
    const isEnsemble = indexType === "ensemble";
    const nprobe = (indexType === "ivfpq" || isEnsemble) ? tuningVal : 64;
    const efSearch = (indexType === "hnswpq" || isEnsemble) ? tuningVal : 64;

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
    if (inFlightController) inFlightController.abort();
    const controller = (inFlightController = new AbortController());
    const seq = ++requestSeq;
    setLoading(true);
    try {
      const response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          query: text,
          top_k: topK,
          nprobe: nprobe,
          index_type: indexType,
          ef_search: efSearch,
          rerank: true,
        }),
      });

      if (!response.ok) throw new Error(await parseErrorResponse(response));

      const data = await response.json();
      if (seq !== requestSeq) return;
      renderResults(data);
    } catch (e) {
      if (e && e.name === "AbortError") return;
      if (seq !== requestSeq) return;
      resultsSection.hidden = true;
      showError(e.message || "Failed to execute text search.");
    } finally {
      if (seq === requestSeq) {
        inFlightController = null;
        setLoading(false);
      }
    }
  }

  async function executeImageSearch(file, topK, nprobe, indexType, efSearch) {
    if (inFlightController) inFlightController.abort();
    const controller = (inFlightController = new AbortController());
    const seq = ++requestSeq;
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
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(await parseErrorResponse(response));

      const data = await response.json();
      if (seq !== requestSeq) return;
      renderResults(data);
    } catch (e) {
      if (e && e.name === "AbortError") return;
      if (seq !== requestSeq) return;
      resultsSection.hidden = true;
      showError(e.message || "Failed to execute image search.");
    } finally {
      if (seq === requestSeq) {
        inFlightController = null;
        setLoading(false);
      }
    }
  }

  // ---------- Render ----------
  function renderResults(data) {
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    let algoBadge;
    if (data.index_type === "ensemble") {
      algoBadge = " [Ensemble: IVF + HNSW, RRF]";
    } else if (data.index_type === "hnswpq") {
      algoBadge = ` [HNSW-PQ, efSearch=${data.ef_search || 64}]`;
    } else {
      algoBadge = ` [IVF-PQ, nprobe=${data.nprobe || 64}]`;
    }
    queryDesc.textContent = (data.query_desc || "Query") + algoBadge;

    metricEmbTime.textContent = `${data.embedding_latency_ms || 0} ms`;
    metricSearchTime.textContent = `${data.search_latency_ms || 0} ms`;
    metricTotalTime.textContent = `${data.total_latency_ms || 0} ms`;

    if (metricFusionWrap && metricFusionTime) {
      // Only a genuine hybrid run has a fusion stage; a non-ensemble
      // response reports 0 and must not claim one.
      if (data.fusion_latency_ms > 0 || data.index_type === "ensemble") {
        metricFusionWrap.hidden = false;
        metricFusionTime.textContent = `${data.fusion_latency_ms || 0} ms`;
      } else {
        metricFusionWrap.hidden = true;
      }
    }

    renderLocationIntel(data);
    renderTopCandidate(data);

    lastQueryMeta = { top_k: data.top_k, nprobe: data.nprobe };
    currentMatches = data.matches || [];
    currentPage = 1;
    renderMatchesPage();

    const identified = (data.voting_summary && data.voting_summary.landmark_name) || "No landmark identified";
    announce(`Search complete. ${identified}. ${pageStatus.textContent}`);
  }

  function renderLocationIntel(data) {
    const loc = data.location_info;
    const hasLoc = loc && (loc.prominent_attraction || loc.city || loc.country);
    if (!hasLoc) {
      locationIntelCard.hidden = true;
      return;
    }

    locationIntelCard.hidden = false;
    locProminentName.textContent = loc.prominent_attraction || "Identified landmark";

    const geoParts = [loc.city, loc.country].filter((p) => p && p !== "nan");
    locGeoText.textContent = geoParts.length > 0 ? geoParts.join(", ") : "World landmark";

    if (loc.specific_subject && loc.specific_subject.toLowerCase() !== (loc.prominent_attraction || "").toLowerCase()) {
      locSpecificWrap.hidden = false;
      locSpecificName.textContent = loc.specific_subject;
    } else {
      locSpecificWrap.hidden = true;
    }

    locSummary.textContent = loc.summary || "";

    // Map
    const gmapsIframe = document.getElementById("gmapsIframe");
    const lat = parseFloat(loc.latitude);
    const lon = parseFloat(loc.longitude);
    let mapQuery = "";

    if (!isNaN(lat) && !isNaN(lon) && (lat !== 0 || lon !== 0)) {
      const ns = lat >= 0 ? "N" : "S";
      const ew = lon >= 0 ? "E" : "W";
      coordsDisplay.textContent = `${Math.abs(lat).toFixed(4)}° ${ns}, ${Math.abs(lon).toFixed(4)}° ${ew}`;
      mapQuery = `${lat},${lon}`;
    } else if (loc.prominent_attraction) {
      coordsDisplay.textContent = "Coordinates estimated";
      mapQuery = encodeURIComponent(`${loc.prominent_attraction} ${loc.city || ""}`.trim());
    } else {
      coordsDisplay.textContent = "";
    }

    if (gmapsIframe) {
      if (mapQuery) {
        gmapsIframe.src = `https://maps.google.com/maps?q=${mapQuery}&t=m&z=15&output=embed&iwloc=near`;
        gmapsIframe.style.display = "";
      } else {
        // Clear the previous query's pin instead of leaving a stale map.
        gmapsIframe.removeAttribute("src");
        gmapsIframe.style.display = "none";
      }
    }

    renderReferenceGallery(data.reference_images || []);
  }

  function renderReferenceGallery(refs) {
    refGalleryGrid.textContent = "";

    if (refs.length === 0) {
      const empty = document.createElement("span");
      empty.className = "gallery-empty";
      empty.textContent = "No canonical photos found.";
      refGalleryGrid.appendChild(empty);
      return;
    }

    refs.forEach((ref) => {
      const refCard = document.createElement("div");
      refCard.className = "ref-card";

      const imgWrap = document.createElement("div");
      imgWrap.className = "ref-img-wrap";

      const refUrl = safeHttpUrl(ref.thumb_url || ref.original_url);
      if (refUrl) {
        const img = document.createElement("img");
        img.src = refUrl;
        img.alt = ref.title ? String(ref.title) : "";
        img.loading = "lazy";
        imgWrap.appendChild(img);
      }

      const title = document.createElement("span");
      title.className = "ref-title";
      title.textContent = ref.title ? String(ref.title) : "";
      title.title = title.textContent;

      refCard.appendChild(imgWrap);
      refCard.appendChild(title);
      refGalleryGrid.appendChild(refCard);
    });
  }

  function renderTopCandidate(data) {
    const v = data.voting_summary;
    if (!v || !v.landmark_name) {
      topCandidateCard.hidden = true;
      return;
    }

    topCandidateCard.hidden = false;
    candName.textContent = v.landmark_name;
    candId.textContent = v.landmark_id;
    candSplit.textContent = v.split || "train";
    candScore.textContent = v.total_vote_score;
    candMargin.textContent = `Lead +${v.confidence_margin}`;
    candMatches.textContent = `${v.match_count} images`;

    if (v.hierarchical_label && v.hierarchical_label !== "nan") {
      candHierarchyWrap.hidden = false;
      candHierarchy.textContent = v.hierarchical_label;
    } else {
      candHierarchyWrap.hidden = true;
    }

    const wikiUrl = safeHttpUrl(v.category_url);
    if (wikiUrl) {
      candWikiBtn.href = wikiUrl;
      candWikiBtn.hidden = false;
    } else {
      candWikiBtn.removeAttribute("href");
      candWikiBtn.hidden = true;
    }
  }

  // ---------- Matches + pagination ----------
  function totalPages() {
    return Math.max(1, Math.ceil(currentMatches.length / MATCHES_PER_PAGE));
  }

  function renderMatchesPage() {
    matchesGrid.textContent = "";

    const total = currentMatches.length;
    matchesCount.textContent = total === 0
      ? "none found"
      : `${total} retrieved · top_k=${lastQueryMeta.top_k} · nprobe=${lastQueryMeta.nprobe}`;

    if (total === 0) {
      const empty = document.createElement("p");
      empty.className = "gallery-empty";
      empty.textContent = "Nothing in the index looked close enough. Try a different photo or description.";
      matchesGrid.appendChild(empty);
      matchesPagination.hidden = true;
      pageStatus.textContent = "No visual matches.";
      return;
    }

    const pages = totalPages();
    if (currentPage > pages) currentPage = pages;

    const start = (currentPage - 1) * MATCHES_PER_PAGE;
    const slice = currentMatches.slice(start, start + MATCHES_PER_PAGE);
    slice.forEach((m) => matchesGrid.appendChild(buildMatchCard(m)));

    matchesPagination.hidden = pages <= 1;
    renderPaginationControls(pages);

    pageStatus.textContent = `Showing ${start + 1}–${start + slice.length} of ${total} matches`;
  }

  function renderPaginationControls(pages) {
    pagePrev.disabled = currentPage === 1;
    pageNext.disabled = currentPage === pages;

    pageList.textContent = "";
    for (let p = 1; p <= pages; p += 1) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "page-num";
      btn.textContent = String(p);
      btn.setAttribute("aria-label", `Page ${p} of ${pages}`);
      if (p === currentPage) btn.setAttribute("aria-current", "page");
      btn.addEventListener("click", () => goToPage(p));
      pageList.appendChild(btn);
    }
  }

  function goToPage(page) {
    const pages = totalPages();
    const next = Math.min(Math.max(1, page), pages);
    if (next === currentPage) return;

    // Re-rendering replaces the pager buttons, so remember where focus was
    // and put it somewhere sensible afterwards instead of dropping to <body>.
    const active = document.activeElement;
    const hadFocusInPager = Boolean(active && matchesPagination.contains(active));
    const wasPrev = active === pagePrev;
    const wasNext = active === pageNext;

    currentPage = next;
    renderMatchesPage();
    announce(pageStatus.textContent);

    if (hadFocusInPager) {
      if (wasPrev && !pagePrev.disabled) pagePrev.focus();
      else if (wasNext && !pageNext.disabled) pageNext.focus();
      else {
        const current = pageList.querySelector('[aria-current="page"]');
        if (current) current.focus();
      }
    }

    matchesGrid.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  pagePrev.addEventListener("click", () => goToPage(currentPage - 1));
  pageNext.addEventListener("click", () => goToPage(currentPage + 1));

  function buildMatchCard(m) {
    const card = document.createElement("div");
    card.className = "match-card";

    const scoreVal = Number(m.final_score ?? m.score ?? 0);
    const pct = Math.min(100, Math.max(10, Math.round(scoreVal * 150)));

    const isVicinityVerified = Boolean(m.is_vicinity_verified);
    const isConsensus = Boolean(m.consensus);

    // Thumbnail path segments come from the mapping parquet, so each segment
    // must be encoded before it is used in a URL.
    const imgId = String(m.image_id || "");
    const imgSplit = String(m.split || "train");
    const thumbUrl = imgId.length >= 3
      ? `/api/image/${encodeURIComponent(imgSplit)}/${encodeURIComponent(imgId[0])}/${encodeURIComponent(imgId[1])}/${encodeURIComponent(imgId[2])}/${encodeURIComponent(imgId)}.jpg`
      : "";
    const matchWikiUrl = safeHttpUrl(m.category_url);

    const detailRows = [
      ["Landmark ID", m.landmark_id],
      ["Image ID", m.image_id],
      ["Split", m.split],
    ];
    if (m.vicinity_cluster) detailRows.push(["Vicinity", m.vicinity_cluster]);
    if (m.distance_km !== null && m.distance_km !== undefined) detailRows.push(["Distance", `${m.distance_km} km`]);
    if (m.visual_sim !== undefined) {
      const pctOf = (x) => Math.round(Number(x || 0) * 100);
      detailRows.push([
        "Breakdown",
        `Vis ${pctOf(m.visual_sim)}% · Vic ${pctOf(m.vicinity_score)}% · Geo ${pctOf(m.spatial_score)}%`,
      ]);
    }
    if (m.sources && m.sources.length > 0) detailRows.push(["Retrieved by", String(m.sources.join(" + ")).toUpperCase()]);
    if (m.hierarchical_label && m.hierarchical_label !== "nan") detailRows.push(["Category", m.hierarchical_label]);

    const detailHtml = detailRows
      .map((row) => `<div class="spec-row"><dt>${escapeHtml(row[0])}</dt><dd class="mono">${escapeHtml(row[1])}</dd></div>`)
      .join("");

    card.innerHTML = `
      ${thumbUrl ? `<div class="match-thumb-wrap">
        <img class="match-thumb" src="${escapeHtml(thumbUrl)}" alt="${escapeHtml(m.landmark_name)}" loading="lazy" />
      </div>` : ""}
      <div class="match-content">
        <div class="match-top">
          <span class="rank-badge ${m.rank <= 3 ? "rank-top" : ""}">#${escapeHtml(m.rank)}</span>
          ${isVicinityVerified ? `<span class="flag">${ICONS.pin} Coordinate</span>` : ""}
          ${isConsensus ? `<span class="flag">${ICONS.star} Consensus</span>` : ""}
          <span class="score-display">${scoreVal.toFixed(4)}</span>
        </div>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${pct}%"></div></div>
        <div class="match-name">${escapeHtml(m.landmark_name)}</div>
        <details class="disclosure">
          <summary><span class="summary-label">Details</span></summary>
          <div class="disclosure-body">
            <dl class="spec-list compact">${detailHtml}</dl>
            ${matchWikiUrl ? `<a href="${escapeHtml(matchWikiUrl)}" target="_blank" rel="noopener noreferrer" class="match-link">Wikimedia category ↗</a>` : ""}
          </div>
        </details>
      </div>
    `;

    const thumbEl = card.querySelector(".match-thumb");
    if (thumbEl) {
      thumbEl.addEventListener("error", () => {
        const wrap = thumbEl.parentElement;
        if (wrap) wrap.style.display = "none";
      });
    }

    return card;
  }

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Data from the GLDv2 parquet and the Wikimedia API is untrusted. Only
  // absolute http(s) URLs may ever reach an href/src, so that a poisoned
  // record cannot smuggle in a javascript: or data: URL.
  function safeHttpUrl(value) {
    if (!value) return "";
    const raw = String(value).trim();
    try {
      const parsed = new URL(raw, window.location.origin);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
      return parsed.href;
    } catch (_) {
      return "";
    }
  }
});
