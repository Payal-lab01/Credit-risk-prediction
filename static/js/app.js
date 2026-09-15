/**
 * Credit Risk Intelligence Cockpit - Client Controller
 * Manages view switching, API interactions, preset loading,
 * form validations, dynamic risk meters, and benchmark charts.
 */

document.addEventListener("DOMContentLoaded", () => {
  // -----------------------------------------------------------------------
  // Global State & Elements
  // -----------------------------------------------------------------------
  let rocChartInstance = null;
  let ksChartInstance = null;
  let cachedMetrics = null;
  let cibilPresets = [];
  let lcPresets = [];

  // Navigation Tabs
  const navButtons = document.querySelectorAll(".nav-btn");
  const viewPanels = document.querySelectorAll(".view-panel");

  // Health Elements
  const healthDot = document.getElementById("health-dot");
  const healthLabel = document.getElementById("health-label");

  // Toast Container
  const toastContainer = document.getElementById("toast-container");

  // -----------------------------------------------------------------------
  // Tab Switching
  // -----------------------------------------------------------------------
  function switchView(targetViewId) {
    viewPanels.forEach(panel => {
      if (panel.id === targetViewId) {
        panel.classList.add("active");
      } else {
        panel.classList.remove("active");
      }
    });

    navButtons.forEach(btn => {
      if (btn.dataset.target === targetViewId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    window.scrollTo({ top: 0, behavior: "smooth" });

    // If switching to comparison, trigger chart resize/update
    if (targetViewId === "view-comparison" && cachedMetrics) {
      renderComparisonCharts(cachedMetrics);
    }
  }

  navButtons.forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.target));
  });

  // Hero quick-action buttons
  document.getElementById("btn-quick-cibil")?.addEventListener("click", () => switchView("view-cibil"));
  document.getElementById("btn-quick-lc")?.addEventListener("click", () => switchView("view-lendingclub"));
  document.getElementById("btn-quick-compare")?.addEventListener("click", () => switchView("view-comparison"));

  // -----------------------------------------------------------------------
  // Toast Notifications
  // -----------------------------------------------------------------------
  function showToast(message, type = "error") {
    const toast = document.createElement("div");
    toast.className = `toast ${type === "error" ? "toast-error" : "toast-success"}`;
    toast.innerHTML = `
      <span>${type === "error" ? "⚠️" : "✅"}</span>
      <span>${message}</span>
    `;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  // -----------------------------------------------------------------------
  // 1. Health Diagnostics
  // -----------------------------------------------------------------------
  async function checkSystemHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      if (data.status === "healthy") {
        healthDot.className = "status-dot pulse-green";
        healthLabel.textContent = "API & Models Active";
      }
    } catch (err) {
      healthDot.className = "status-dot";
      healthDot.style.background = "#f43f5e";
      healthLabel.textContent = "API Offline";
      console.error("Health check error:", err);
    }
  }

  // -----------------------------------------------------------------------
  // 2. Fetch Authentic Metrics & Populate Overview & Comparison
  // -----------------------------------------------------------------------
  async function loadMetrics() {
    try {
      const res = await fetch("/api/metrics");
      if (!res.ok) throw new Error("Could not load metrics");
      const json = await res.json();
      cachedMetrics = json.data;

      // Overview CIBIL Cards
      const cibilBal = cachedMetrics.cibil.models.logistic_regression_balanced;
      document.getElementById("overview-cibil-auc").textContent = cibilBal.roc_auc.toFixed(4);
      document.getElementById("overview-cibil-ks").textContent = cibilBal.ks_statistic.toFixed(4);
      document.getElementById("overview-cibil-recall").textContent = (cibilBal.recall * 100).toFixed(2) + "%";
      document.getElementById("overview-cibil-prec").textContent = (cibilBal.precision * 100).toFixed(2) + "%";

      // Overview LendingClub Cards
      const lcLog = cachedMetrics.lending_club.models.logistic_regression;
      const lcRf = cachedMetrics.lending_club.models.random_forest;
      document.getElementById("overview-lc-auc").textContent = lcLog.roc_auc.toFixed(4);
      document.getElementById("overview-lc-ks").textContent = lcLog.ks_statistic.toFixed(4);
      document.getElementById("overview-lc-thresh").textContent = lcLog.threshold.toFixed(2);
      document.getElementById("overview-lc-recall").textContent = (lcLog.recall * 100).toFixed(2) + "%";

      document.getElementById("overview-rf-auc").textContent = lcRf.roc_auc.toFixed(4);
      document.getElementById("overview-rf-ks").textContent = lcRf.ks_statistic.toFixed(4);
      document.getElementById("overview-rf-recall").textContent = (lcRf.recall * 100).toFixed(2) + "%";

      // Populate Table in Comparison View
      populateMetricsTable(cachedMetrics);

      // Render Charts
      renderComparisonCharts(cachedMetrics);
    } catch (err) {
      console.error("Error loading metrics:", err);
    }
  }

  function populateMetricsTable(metrics) {
    const tbody = document.getElementById("metrics-table-body");
    if (!tbody) return;

    const cibilStd = metrics.cibil.models.logistic_regression;
    const cibilBal = metrics.cibil.models.logistic_regression_balanced;
    const lcLog = metrics.lending_club.models.logistic_regression;
    const lcRf = metrics.lending_club.models.random_forest;

    tbody.innerHTML = `
      <tr>
        <td>🇮🇳 CIBIL (India)</td>
        <td>Logistic Regression (Standard)</td>
        <td><strong>${cibilStd.roc_auc.toFixed(4)}</strong></td>
        <td>${cibilStd.ks_statistic.toFixed(4)}</td>
        <td>${(cibilStd.recall * 100).toFixed(2)}%</td>
        <td>${(cibilStd.precision * 100).toFixed(2)}%</td>
        <td>${cibilStd.threshold.toFixed(2)}</td>
        <td><span class="badge badge-gray">Notebook Baseline</span></td>
      </tr>
      <tr>
        <td>🇮🇳 CIBIL (India)</td>
        <td>Logistic Regression (Balanced)</td>
        <td><strong>${cibilBal.roc_auc.toFixed(4)}</strong></td>
        <td><strong>${cibilBal.ks_statistic.toFixed(4)}</strong></td>
        <td><strong>${(cibilBal.recall * 100).toFixed(2)}%</strong></td>
        <td><strong>${(cibilBal.precision * 100).toFixed(2)}%</strong></td>
        <td>${cibilBal.threshold.toFixed(2)}</td>
        <td><span class="badge badge-green">Production Deployed</span></td>
      </tr>
      <tr>
        <td>🇺🇸 LendingClub (US)</td>
        <td>Calibrated Logistic Regression</td>
        <td><strong>${lcLog.roc_auc.toFixed(4)}</strong></td>
        <td>${lcLog.ks_statistic.toFixed(4)}</td>
        <td><strong>${(lcLog.recall * 100).toFixed(2)}%</strong></td>
        <td>${(lcLog.precision * 100).toFixed(2)}%</td>
        <td><span class="badge badge-amber">${lcLog.threshold.toFixed(2)} (Calibrated)</span></td>
        <td><span class="badge badge-green">Production Deployed</span></td>
      </tr>
      <tr>
        <td>🇺🇸 LendingClub (US)</td>
        <td>Random Forest Classifier</td>
        <td><strong>${lcRf.roc_auc.toFixed(4)}</strong></td>
        <td>${lcRf.ks_statistic.toFixed(4)}</td>
        <td><strong>${(lcRf.recall * 100).toFixed(2)}%</strong></td>
        <td>${(lcRf.precision * 100).toFixed(2)}%</td>
        <td>${lcRf.threshold.toFixed(2)}</td>
        <td><span class="badge badge-green">Production Deployed</span></td>
      </tr>
    `;
  }

  function renderComparisonCharts(metrics) {
    if (typeof Chart === "undefined") return;

    const cibilBal = metrics.cibil.models.logistic_regression_balanced;
    const lcLog = metrics.lending_club.models.logistic_regression;
    const lcRf = metrics.lending_club.models.random_forest;

    const labels = [
      "CIBIL Logistic (India)",
      "LC Logistic (US)",
      "LC Random Forest (US)"
    ];

    // 1. ROC-AUC Chart
    const rocCtx = document.getElementById("chart-roc-auc")?.getContext("2d");
    if (rocCtx) {
      if (rocChartInstance) rocChartInstance.destroy();
      rocChartInstance = new Chart(rocCtx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "ROC-AUC Score",
            data: [cibilBal.roc_auc, lcLog.roc_auc, lcRf.roc_auc],
            backgroundColor: ["#6366f1", "#06b6d4", "#10b981"],
            borderRadius: 8,
            barThickness: 42,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => ` ROC-AUC: ${ctx.parsed.y.toFixed(4)}`
              }
            }
          },
          scales: {
            y: {
              min: 0.5,
              max: 1.0,
              grid: { color: "rgba(255, 255, 255, 0.06)" },
              ticks: { color: "#94a3b8" }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#e2e8f0", font: { size: 11 } }
            }
          }
        }
      });
    }

    // 2. KS Statistic Chart
    const ksCtx = document.getElementById("chart-ks")?.getContext("2d");
    if (ksCtx) {
      if (ksChartInstance) ksChartInstance.destroy();
      ksChartInstance = new Chart(ksCtx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "KS Separation Statistic",
            data: [cibilBal.ks_statistic, lcLog.ks_statistic, lcRf.ks_statistic],
            backgroundColor: ["#818cf8", "#22d3ee", "#34d399"],
            borderRadius: 8,
            barThickness: 42,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => ` KS Statistic: ${ctx.parsed.y.toFixed(4)}`
              }
            }
          },
          scales: {
            y: {
              min: 0.0,
              max: 0.7,
              grid: { color: "rgba(255, 255, 255, 0.06)" },
              ticks: { color: "#94a3b8" }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#e2e8f0", font: { size: 11 } }
            }
          }
        }
      });
    }
  }

  // -----------------------------------------------------------------------
  // 3. Presets Management
  // -----------------------------------------------------------------------
  async function loadPresets() {
    try {
      const [resCibil, resLc] = await Promise.all([
        fetch("/api/presets/cibil"),
        fetch("/api/presets/lendingclub")
      ]);
      if (resCibil.ok) {
        const d = await resCibil.json();
        cibilPresets = d.presets;
      }
      if (resLc.ok) {
        const d = await resLc.json();
        lcPresets = d.presets;
      }
    } catch (err) {
      console.error("Error loading presets:", err);
    }
  }

  // Bind CIBIL Presets
  document.getElementById("btn-cibil-preset-prime")?.addEventListener("click", () => {
    const p = cibilPresets.find(x => x.id === "cibil_prime");
    if (p) populateCibilForm(p.data);
  });
  document.getElementById("btn-cibil-preset-near")?.addEventListener("click", () => {
    const p = cibilPresets.find(x => x.id === "cibil_near_prime");
    if (p) populateCibilForm(p.data);
  });
  document.getElementById("btn-cibil-preset-sub")?.addEventListener("click", () => {
    const p = cibilPresets.find(x => x.id === "cibil_subprime");
    if (p) populateCibilForm(p.data);
  });
  document.getElementById("btn-cibil-reset")?.addEventListener("click", () => {
    document.getElementById("form-cibil").reset();
    updateScoreDisplay(720);
    resetCibilResult();
  });

  // Bind LendingClub Presets
  document.getElementById("btn-lc-preset-prime")?.addEventListener("click", () => {
    const p = lcPresets.find(x => x.id === "lc_prime");
    if (p) populateLcForm(p.data);
  });
  document.getElementById("btn-lc-preset-mid")?.addEventListener("click", () => {
    const p = lcPresets.find(x => x.id === "lc_moderate");
    if (p) populateLcForm(p.data);
  });
  document.getElementById("btn-lc-preset-sub")?.addEventListener("click", () => {
    const p = lcPresets.find(x => x.id === "lc_subprime");
    if (p) populateLcForm(p.data);
  });
  document.getElementById("btn-lc-reset")?.addEventListener("click", () => {
    document.getElementById("form-lendingclub").reset();
    syncSubgrades("B", "B1");
    resetLcResult();
  });

  // -----------------------------------------------------------------------
  // 4. CIBIL Form & Evaluation
  // -----------------------------------------------------------------------
  const cibilScoreSlider = document.getElementById("cibil-score");
  const scoreDisplay = document.getElementById("score-display");
  const scoreBadge = document.getElementById("score-badge");

  function updateScoreDisplay(val) {
    const score = parseInt(val, 10);
    scoreDisplay.textContent = score;
    if (score >= 750) {
      scoreBadge.className = "score-badge badge-good";
      scoreBadge.textContent = "Prime / Good Standing";
    } else if (score >= 650) {
      scoreBadge.className = "score-badge badge-fair";
      scoreBadge.textContent = "Near Prime / Moderate";
    } else {
      scoreBadge.className = "score-badge badge-poor";
      scoreBadge.textContent = "Subprime / Elevated Risk";
    }
  }

  cibilScoreSlider?.addEventListener("input", e => updateScoreDisplay(e.target.value));

  function populateCibilForm(data) {
    for (const [key, value] of Object.entries(data)) {
      const el = document.querySelector(`[name="${key}"]`);
      if (el) el.value = value;
    }
    if (data.Credit_Score) updateScoreDisplay(data.Credit_Score);
  }

  function resetCibilResult() {
    document.getElementById("cibil-placeholder").classList.remove("hidden");
    document.getElementById("cibil-content").classList.add("hidden");
  }

  const formCibil = document.getElementById("form-cibil");
  formCibil?.addEventListener("submit", async e => {
    e.preventDefault();
    const btn = document.getElementById("btn-cibil-submit");
    const spinner = document.getElementById("cibil-spinner");

    btn.disabled = true;
    spinner.classList.remove("hidden");

    try {
      const formData = new FormData(formCibil);
      const payload = {};
      for (const [k, v] of formData.entries()) {
        payload[k] = isNaN(v) || v === "" ? v : parseFloat(v);
      }

      // Check optional overrides
      const overrides = {};
      const advPmnt = document.getElementById("adv-time-pmnt")?.value;
      if (advPmnt) overrides["time_since_recent_payment"] = parseFloat(advPmnt);
      const advDelinq = document.getElementById("adv-max-delinq")?.value;
      if (advDelinq) overrides["max_delinquency_level"] = parseFloat(advDelinq);
      const adv30 = document.getElementById("adv-30p-dpd")?.value;
      if (adv30) overrides["num_times_30p_dpd"] = parseFloat(adv30);
      const adv60 = document.getElementById("adv-60p-dpd")?.value;
      if (adv60) overrides["num_times_60p_dpd"] = parseFloat(adv60);
      const advExp = document.getElementById("adv-unsec-exp")?.value;
      if (advExp) overrides["max_unsec_exposure_inPct"] = parseFloat(advExp);
      const advOld = document.getElementById("adv-oldest-tl")?.value;
      if (advOld) overrides["Age_Oldest_TL"] = parseFloat(advOld);

      payload["advanced_bureau_overrides"] = overrides;

      const res = await fetch("/api/predict/cibil", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || "Failed to calculate CIBIL credit risk.");
      }

      renderCibilResult(data);
      showToast("CIBIL prediction calculated successfully.", "success");

      // Non-blocking asynchronous explainability fetch
      fetchAndRenderCibilExplanation(payload, data);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btn.disabled = false;
      spinner.classList.add("hidden");
    }
  });

  function renderCibilResult(data) {
    document.getElementById("cibil-placeholder").classList.add("hidden");
    document.getElementById("cibil-content").classList.remove("hidden");

    // Action Badge
    const actionBadge = document.getElementById("cibil-action-badge");
    if (data.action === "Approve") {
      actionBadge.className = "result-badge badge-approved";
      actionBadge.textContent = "APPROVED";
    } else if (data.action === "Manual Review") {
      actionBadge.className = "result-badge badge-review";
      actionBadge.textContent = "MANUAL REVIEW";
    } else {
      actionBadge.className = "result-badge badge-decline";
      actionBadge.textContent = "DECLINE / HIGH RISK";
    }

    // Gauge calculation (125.66 is half circumference)
    const prob = data.default_probability;
    const offset = 125.66 * (1 - prob);
    const gaugeFill = document.getElementById("cibil-gauge-fill");
    gaugeFill.style.strokeDashoffset = offset;

    if (prob < 0.35) {
      gaugeFill.style.stroke = "#10b981";
    } else if (prob < 0.50) {
      gaugeFill.style.stroke = "#f59e0b";
    } else {
      gaugeFill.style.stroke = "#f43f5e";
    }

    document.getElementById("cibil-prob-percent").textContent = (prob * 100).toFixed(1) + "%";
    document.getElementById("cibil-prob-val").textContent = prob.toFixed(4);
    document.getElementById("cibil-approval-val").textContent = (data.approval_probability * 100).toFixed(1) + "%";
    document.getElementById("cibil-tier-val").textContent = data.risk_tier;

    // Rationale text
    const rationale = document.getElementById("cibil-rationale-text");
    if (data.prediction === 0) {
      rationale.textContent = `Default probability of ${(prob * 100).toFixed(1)}% is comfortably below the 50.0% threshold. CIBIL score (${data.summary_features.Credit_Score}) and trade-line performance support prime approval.`;
    } else {
      rationale.textContent = `Default probability of ${(prob * 100).toFixed(1)}% breaches the risk tolerance threshold. Applicant exhibits elevated risk signals across historical missed payments (${data.summary_features.Tot_Missed_Pmnt}) or bureau inquiries.`;
    }
  }

  // -----------------------------------------------------------------------
  // 5. LendingClub Dynamic Sub-Grades & Evaluation
  // -----------------------------------------------------------------------
  const lcGradeSelect = document.getElementById("lc-grade");
  const lcSubgradeSelect = document.getElementById("lc-subgrade");

  function syncSubgrades(selectedGrade, preferredSubgrade = null) {
    lcSubgradeSelect.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const sub = `${selectedGrade}${i}`;
      const opt = document.createElement("option");
      opt.value = sub;
      opt.textContent = sub;
      if (preferredSubgrade && preferredSubgrade === sub) opt.selected = true;
      lcSubgradeSelect.appendChild(opt);
    }
  }

  lcGradeSelect?.addEventListener("change", e => syncSubgrades(e.target.value));

  function populateLcForm(data) {
    for (const [key, value] of Object.entries(data)) {
      const el = document.querySelector(`[name="${key}"]`);
      if (el) el.value = value;
    }
    if (data.grade) {
      syncSubgrades(data.grade, data.sub_grade || `${data.grade}1`);
    }
  }

  function resetLcResult() {
    document.getElementById("lc-placeholder").classList.remove("hidden");
    document.getElementById("lc-content").classList.add("hidden");
  }

  const formLc = document.getElementById("form-lendingclub");
  formLc?.addEventListener("submit", async e => {
    e.preventDefault();
    const btn = document.getElementById("btn-lc-submit");
    const spinner = document.getElementById("lc-spinner");

    btn.disabled = true;
    spinner.classList.remove("hidden");

    try {
      const formData = new FormData(formLc);
      const payload = {};
      for (const [k, v] of formData.entries()) {
        payload[k] = isNaN(v) || v === "" ? v : parseFloat(v);
      }

      const res = await fetch("/api/predict/lendingclub", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || "Failed to calculate LendingClub credit risk.");
      }

      renderLcResult(data);
      showToast("LendingClub dual-model assessment complete.", "success");

      // Non-blocking asynchronous explainability fetch
      fetchAndRenderLcExplanation(payload, data);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btn.disabled = false;
      spinner.classList.add("hidden");
    }
  });

  function renderLcResult(data) {
    document.getElementById("lc-placeholder").classList.add("hidden");
    document.getElementById("lc-content").classList.remove("hidden");

    const logreg = data.evaluated_models.logistic_regression;
    const rf = data.evaluated_models.random_forest;
    const consensus = data.primary_recommendation;

    // Consensus Banner
    const banner = document.getElementById("lc-consensus-banner");
    const bannerTitle = document.getElementById("lc-consensus-title");
    const bannerDesc = document.getElementById("lc-consensus-desc");

    bannerTitle.textContent = consensus.consensus.toUpperCase();
    if (consensus.action === "Approve") {
      banner.style.background = "rgba(16, 185, 129, 0.15)";
      banner.style.borderColor = "rgba(16, 185, 129, 0.35)";
      bannerTitle.style.color = "#34d399";
      bannerDesc.textContent = "Both models concur on low default likelihood.";
    } else if (consensus.action === "Decline") {
      banner.style.background = "rgba(244, 63, 94, 0.15)";
      banner.style.borderColor = "rgba(244, 63, 94, 0.35)";
      bannerTitle.style.color = "#fb7185";
      bannerDesc.textContent = "Both models indicate critical default vulnerability.";
    } else {
      banner.style.background = "rgba(245, 158, 11, 0.15)";
      banner.style.borderColor = "rgba(245, 158, 11, 0.35)";
      bannerTitle.style.color = "#fbbf24";
      bannerDesc.textContent = "Calibrated Logistic threshold triggered; manual credit review advised.";
    }

    // Model 1: Calibrated Logistic
    document.getElementById("lc-logreg-prob").textContent = (logreg.default_probability * 100).toFixed(1) + "%";
    const logBadge = document.getElementById("lc-logreg-badge");
    if (logreg.prediction === 0) {
      logBadge.className = "eval-badge badge-approved";
      logBadge.textContent = "APPROVED";
    } else {
      logBadge.className = "eval-badge badge-decline";
      logBadge.textContent = "DECLINE";
    }
    const logBarWidth = Math.min(100, Math.max(2, logreg.default_probability * 100));
    const logBar = document.getElementById("lc-logreg-bar");
    logBar.style.width = logBarWidth + "%";
    logBar.style.background = logreg.prediction === 0 ? "#06b6d4" : "#f43f5e";

    document.getElementById("lc-logreg-note").textContent =
      `Default probability ${(logreg.default_probability * 100).toFixed(1)}% evaluated against calibrated 13.0% threshold.`;

    // Model 2: Random Forest
    document.getElementById("lc-rf-prob").textContent = (rf.default_probability * 100).toFixed(1) + "%";
    const rfBadge = document.getElementById("lc-rf-badge");
    if (rf.prediction === 0) {
      rfBadge.className = "eval-badge badge-approved";
      rfBadge.textContent = "APPROVED";
    } else {
      rfBadge.className = "eval-badge badge-decline";
      rfBadge.textContent = "DECLINE";
    }
    const rfBarWidth = Math.min(100, Math.max(2, rf.default_probability * 100));
    const rfBar = document.getElementById("lc-rf-bar");
    rfBar.style.width = rfBarWidth + "%";
    rfBar.style.background = rf.prediction === 0 ? "#10b981" : "#f43f5e";

    document.getElementById("lc-rf-note").textContent =
      `Ensemble vote ${(rf.default_probability * 100).toFixed(1)}% evaluated on unscaled features at 50.0% threshold.`;
  }

  // -----------------------------------------------------------------------
  // 6. Explainability Controllers (SHAP & LIME)
  // -----------------------------------------------------------------------
  let currentCibilExplain = null;
  let cibilMethod = "shap"; // 'shap' | 'lime'

  let currentLcExplain = null;
  let lcSelectedModel = "logistic"; // 'logistic' | 'rf'
  let lcMethod = "shap"; // 'shap' | 'lime'

  // --- CIBIL Explainability Fetch & Render ---
  async function fetchAndRenderCibilExplanation(payload, predData) {
    const sub = document.getElementById("cibil-explain-model-sub");
    if (sub) sub.textContent = "Computing SHAP & LIME explanations...";

    try {
      const res = await fetch("/api/explain/cibil", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const expData = await res.json();
      if (!res.ok) throw new Error(expData.message || "Failed to generate CIBIL explanation.");

      currentCibilExplain = expData;
      renderCibilExplainView();
    } catch (err) {
      console.warn("CIBIL explainability notice:", err);
      const bars = document.getElementById("cibil-contrib-bars");
      if (bars) {
        bars.innerHTML = `<div style="font-size:0.8rem; color:#f59e0b; padding:8px;">⚠️ Explainability notice: ${err.message}</div>`;
      }
    }
  }

  function renderCibilExplainView() {
    if (!currentCibilExplain || currentCibilExplain.status !== "success") return;

    // Subtitle
    const prob = (currentCibilExplain.default_probability * 100).toFixed(1);
    document.getElementById("cibil-explain-model-sub").textContent =
      `${currentCibilExplain.model_name} | Default Prob: ${prob}% (${currentCibilExplain.risk_label})`;

    const posList = document.getElementById("cibil-pos-list");
    const negList = document.getElementById("cibil-neg-list");
    const barsContainer = document.getElementById("cibil-contrib-bars");
    const chartTitle = document.getElementById("cibil-chart-method-title");

    posList.innerHTML = "";
    negList.innerHTML = "";
    barsContainer.innerHTML = "";

    if (cibilMethod === "shap") {
      chartTitle.textContent = "SHAP Feature Attribution (Log-Odds Impact)";

      // Top approval factors
      const posFactors = currentCibilExplain.shap.top_approval_factors;
      if (posFactors.length === 0) {
        posList.innerHTML = "<li>No significant approval factors identified.</li>";
      } else {
        posFactors.forEach(f => {
          const li = document.createElement("li");
          li.textContent = f.human_text;
          posList.appendChild(li);
        });
      }

      // Top risk factors
      const negFactors = currentCibilExplain.shap.top_risk_factors;
      if (negFactors.length === 0) {
        negList.innerHTML = "<li>No significant risk factors identified.</li>";
      } else {
        negFactors.forEach(f => {
          const li = document.createElement("li");
          li.textContent = f.human_text;
          negList.appendChild(li);
        });
      }

      // Horizontal contribution bars
      const factors = currentCibilExplain.shap.all_top_factors;
      const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);

      factors.forEach(f => {
        const isApprove = f.shap_value < 0;
        const widthPct = Math.min(100, Math.max(8, (Math.abs(f.shap_value) / maxAbs) * 100));
        const barClass = isApprove ? "contrib-bar-approval" : "contrib-bar-risk";
        const valClass = isApprove ? "val-approval" : "val-risk";
        const badgeClass = f.source === "user_input" ? "badge-source-user" : "badge-source-base";
        const badgeLabel = f.source === "user_input" ? "User" : "Baseline";

        const row = document.createElement("div");
        row.className = "contrib-row";
        row.innerHTML = `
          <div class="contrib-feat-col" title="${f.feature} = ${f.raw_value}">
            <span class="${badgeClass}">${badgeLabel}</span>
            <span class="contrib-feat-name">${f.display_name}</span>
          </div>
          <div class="contrib-track">
            <div class="contrib-bar-fill ${barClass}" style="width: ${widthPct}%;"></div>
          </div>
          <div class="contrib-val ${valClass}">
            ${f.shap_value > 0 ? "+" : ""}${f.shap_value.toFixed(2)}
          </div>
        `;
        barsContainer.appendChild(row);
      });
    } else {
      // LIME View
      chartTitle.textContent = "LIME Local Explanation (Perturbation Weights)";
      const limeFactors = currentCibilExplain.lime.factors;
      const maxAbs = Math.max(...limeFactors.map(f => Math.abs(f.lime_weight)), 0.001);

      const posLimes = limeFactors.filter(f => f.lime_weight < 0);
      const negLimes = limeFactors.filter(f => f.lime_weight > 0);

      posLimes.slice(0, 3).forEach(f => {
        const li = document.createElement("li");
        li.textContent = f.human_text;
        posList.appendChild(li);
      });
      if (posLimes.length === 0) posList.innerHTML = "<li>No significant approval factors.</li>";

      negLimes.slice(0, 3).forEach(f => {
        const li = document.createElement("li");
        li.textContent = f.human_text;
        negList.appendChild(li);
      });
      if (negLimes.length === 0) negList.innerHTML = "<li>No significant risk factors.</li>";

      limeFactors.forEach(f => {
        const isApprove = f.lime_weight < 0;
        const widthPct = Math.min(100, Math.max(8, (Math.abs(f.lime_weight) / maxAbs) * 100));
        const barClass = isApprove ? "contrib-bar-approval" : "contrib-bar-risk";
        const valClass = isApprove ? "val-approval" : "val-risk";
        const badgeClass = f.source === "user_input" ? "badge-source-user" : "badge-source-base";
        const badgeLabel = f.source === "user_input" ? "User" : "Baseline";

        const row = document.createElement("div");
        row.className = "contrib-row";
        row.innerHTML = `
          <div class="contrib-feat-col" title="${f.feature}">
            <span class="${badgeClass}">${badgeLabel}</span>
            <span class="contrib-feat-name">${f.display_name}</span>
          </div>
          <div class="contrib-track">
            <div class="contrib-bar-fill ${barClass}" style="width: ${widthPct}%;"></div>
          </div>
          <div class="contrib-val ${valClass}">
            ${f.lime_weight > 0 ? "+" : ""}${f.lime_weight.toFixed(3)}
          </div>
        `;
        barsContainer.appendChild(row);
      });
    }
  }

  // Bind CIBIL Method Buttons
  document.getElementById("cibil-btn-shap")?.addEventListener("click", () => {
    cibilMethod = "shap";
    document.getElementById("cibil-btn-shap").classList.add("active");
    document.getElementById("cibil-btn-lime").classList.remove("active");
    renderCibilExplainView();
  });
  document.getElementById("cibil-btn-lime")?.addEventListener("click", () => {
    cibilMethod = "lime";
    document.getElementById("cibil-btn-lime").classList.add("active");
    document.getElementById("cibil-btn-shap").classList.remove("active");
    renderCibilExplainView();
  });

  // --- LendingClub Explainability Fetch & Render ---
  async function fetchAndRenderLcExplanation(payload, predData) {
    const sub = document.getElementById("lc-explain-model-sub");
    if (sub) sub.textContent = "Computing dual-model SHAP & LIME explanations...";

    try {
      const res = await fetch("/api/explain/lendingclub", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const expData = await res.json();
      if (!res.ok) throw new Error(expData.message || "Failed to generate LendingClub explanation.");

      currentLcExplain = expData;
      renderLcExplainView();
    } catch (err) {
      console.warn("LendingClub explainability notice:", err);
      const bars = document.getElementById("lc-contrib-bars");
      if (bars) {
        bars.innerHTML = `<div style="font-size:0.8rem; color:#f59e0b; padding:8px;">⚠️ Explainability notice: ${err.message}</div>`;
      }
    }
  }

  function renderLcExplainView() {
    if (!currentLcExplain || currentLcExplain.status !== "success") return;

    const modelKey = lcSelectedModel === "logistic" ? "logistic_regression" : "random_forest";
    const modelData = currentLcExplain[modelKey];

    // Subtitle & Divergence Text
    const prob = (modelData.default_probability * 100).toFixed(1);
    document.getElementById("lc-explain-model-sub").textContent =
      `${modelData.model_name} | Default Prob: ${prob}% (${modelData.risk_label})`;

    const divText = document.getElementById("lc-divergence-text");
    if (lcSelectedModel === "logistic") {
      divText.textContent =
        "Logistic Regression calculates linear log-odds across scaled continuous features (e.g. interest rate, DTI, revolving utilization) with binary grade indicators.";
    } else {
      divText.textContent =
        "Random Forest splits trees on non-linear loan grade interactions, borrower inquiries, and debt-to-income thresholds using raw unscaled metrics.";
    }

    const posList = document.getElementById("lc-pos-list");
    const negList = document.getElementById("lc-neg-list");
    const barsContainer = document.getElementById("lc-contrib-bars");
    const chartTitle = document.getElementById("lc-chart-method-title");

    posList.innerHTML = "";
    negList.innerHTML = "";
    barsContainer.innerHTML = "";

    if (lcMethod === "shap") {
      chartTitle.textContent = `SHAP Feature Attribution (${modelData.model_name})`;
      const posFactors = modelData.shap.top_approval_factors;
      if (posFactors.length === 0) {
        posList.innerHTML = "<li>No significant approval factors.</li>";
      } else {
        posFactors.forEach(f => {
          const li = document.createElement("li");
          li.textContent = f.human_text;
          posList.appendChild(li);
        });
      }

      const negFactors = modelData.shap.top_risk_factors;
      if (negFactors.length === 0) {
        negList.innerHTML = "<li>No significant risk factors.</li>";
      } else {
        negFactors.forEach(f => {
          const li = document.createElement("li");
          li.textContent = f.human_text;
          negList.appendChild(li);
        });
      }

      const factors = modelData.shap.all_top_factors;
      const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);

      factors.forEach(f => {
        const isApprove = f.shap_value < 0;
        const widthPct = Math.min(100, Math.max(8, (Math.abs(f.shap_value) / maxAbs) * 100));
        const barClass = isApprove ? "contrib-bar-approval" : "contrib-bar-risk";
        const valClass = isApprove ? "val-approval" : "val-risk";

        const row = document.createElement("div");
        row.className = "contrib-row";
        row.innerHTML = `
          <div class="contrib-feat-col" title="${f.feature} = ${f.raw_value}">
            <span class="contrib-feat-name">${f.display_name}</span>
          </div>
          <div class="contrib-track">
            <div class="contrib-bar-fill ${barClass}" style="width: ${widthPct}%;"></div>
          </div>
          <div class="contrib-val ${valClass}">
            ${f.shap_value > 0 ? "+" : ""}${f.shap_value.toFixed(3)}
          </div>
        `;
        barsContainer.appendChild(row);
      });
    } else {
      // LIME View
      chartTitle.textContent = `LIME Local Explanation (${modelData.model_name})`;
      const limeFactors = modelData.lime.factors;
      const maxAbs = Math.max(...limeFactors.map(f => Math.abs(f.lime_weight)), 0.001);

      const posLimes = limeFactors.filter(f => f.lime_weight < 0);
      const negLimes = limeFactors.filter(f => f.lime_weight > 0);

      posLimes.slice(0, 3).forEach(f => {
        const li = document.createElement("li");
        li.textContent = f.human_text;
        posList.appendChild(li);
      });
      if (posLimes.length === 0) posList.innerHTML = "<li>No significant approval factors.</li>";

      negLimes.slice(0, 3).forEach(f => {
        const li = document.createElement("li");
        li.textContent = f.human_text;
        negList.appendChild(li);
      });
      if (negLimes.length === 0) negList.innerHTML = "<li>No significant risk factors.</li>";

      limeFactors.forEach(f => {
        const isApprove = f.lime_weight < 0;
        const widthPct = Math.min(100, Math.max(8, (Math.abs(f.lime_weight) / maxAbs) * 100));
        const barClass = isApprove ? "contrib-bar-approval" : "contrib-bar-risk";
        const valClass = isApprove ? "val-approval" : "val-risk";

        const row = document.createElement("div");
        row.className = "contrib-row";
        row.innerHTML = `
          <div class="contrib-feat-col" title="${f.feature}">
            <span class="contrib-feat-name">${f.display_name}</span>
          </div>
          <div class="contrib-track">
            <div class="contrib-bar-fill ${barClass}" style="width: ${widthPct}%;"></div>
          </div>
          <div class="contrib-val ${valClass}">
            ${f.lime_weight > 0 ? "+" : ""}${f.lime_weight.toFixed(3)}
          </div>
        `;
        barsContainer.appendChild(row);
      });
    }
  }

  // Bind LendingClub Model and Method Buttons
  document.getElementById("lc-btn-model-log")?.addEventListener("click", () => {
    lcSelectedModel = "logistic";
    document.getElementById("lc-btn-model-log").classList.add("active");
    document.getElementById("lc-btn-model-rf").classList.remove("active");
    renderLcExplainView();
  });
  document.getElementById("lc-btn-model-rf")?.addEventListener("click", () => {
    lcSelectedModel = "rf";
    document.getElementById("lc-btn-model-rf").classList.add("active");
    document.getElementById("lc-btn-model-log").classList.remove("active");
    renderLcExplainView();
  });
  document.getElementById("lc-btn-shap")?.addEventListener("click", () => {
    lcMethod = "shap";
    document.getElementById("lc-btn-shap").classList.add("active");
    document.getElementById("lc-btn-lime").classList.remove("active");
    renderLcExplainView();
  });
  document.getElementById("lc-btn-lime")?.addEventListener("click", () => {
    lcMethod = "lime";
    document.getElementById("lc-btn-lime").classList.add("active");
    document.getElementById("lc-btn-shap").classList.remove("active");
    renderLcExplainView();
  });

  // -----------------------------------------------------------------------
  // Initial Boot
  // -----------------------------------------------------------------------
  checkSystemHealth();
  loadMetrics();
  loadPresets();
  syncSubgrades("B", "B3");
});

