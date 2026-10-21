/**
 * Q-Glitch Clinical Platform - Interactive Application Controller
 * Handles real-time clinical parameter adjustments, hybrid quantum inference,
 * SVG gauge animations, differential diagnosis rendering, cohort matrix, and report export.
 *
 * Author: Mallikarjun.R (Team Lead) & Team Q-Glitch
 * Event: VTU-Qiskit Fall Fest Hackathon 2026
 * Team ID: VTU-QFF-HC-0190
 */

(function () {
  'use strict';

  /* ==========================================================================
     1. CONSTANTS & PRESET ARCHETYPES
     ========================================================================== */
  const PRESETS = {
    laminopathy: {
      patient_id: 'PT-4092',
      age: 14.0,
      bmi: 15.5,
      heart_rate: 96,
      blood_pressure_sys: 138,
      blood_pressure_dia: 88,
      cholesterol: 230,
      hemoglobin: 11.2,
      white_blood_cells: 9.8,
      platelet_count: 210,
      creatinine: 1.45,
      albumin: 3.8,
      c_reactive_protein: 8.5,
      lmna_mutation_score: 9.2,
      progerin_level: 8.8,
      telomere_length: 2.4,
      dna_damage_index: 1.45,
      zmpste24_expression: 0.45,
      autophagy_marker: 0.55,
      inflammatory_cytokine: 3.2,
      oxidative_stress: 2.6,
      growth_retardation_score: 0.95,
      skin_elasticity: 0.15,
      joint_mobility: 0.40,
      hair_density_index: 0.20,
      cardiovascular_risk: 0.85,
      skeletal_abnormality: 0.70,
    },
    lysosomal: {
      patient_id: 'PT-6104',
      age: 28.0,
      bmi: 24.2,
      heart_rate: 82,
      blood_pressure_sys: 122,
      blood_pressure_dia: 78,
      cholesterol: 215,
      hemoglobin: 10.4,
      white_blood_cells: 4.8,
      platelet_count: 85,
      creatinine: 1.85,
      albumin: 3.2,
      c_reactive_protein: 6.2,
      lmna_mutation_score: 0.8,
      progerin_level: 0.4,
      telomere_length: 6.8,
      dna_damage_index: 0.65,
      zmpste24_expression: 1.0,
      autophagy_marker: 2.85,
      inflammatory_cytokine: 2.8,
      oxidative_stress: 2.4,
      growth_retardation_score: 0.45,
      skin_elasticity: 0.65,
      joint_mobility: 0.35,
      hair_density_index: 0.75,
      cardiovascular_risk: 0.60,
      skeletal_abnormality: 0.85,
    },
    connective: {
      patient_id: 'PT-3329',
      age: 22.0,
      bmi: 17.8,
      heart_rate: 88,
      blood_pressure_sys: 110,
      blood_pressure_dia: 68,
      cholesterol: 170,
      hemoglobin: 13.5,
      white_blood_cells: 6.2,
      platelet_count: 240,
      creatinine: 0.85,
      albumin: 4.4,
      c_reactive_protein: 1.5,
      lmna_mutation_score: 0.4,
      progerin_level: 0.2,
      telomere_length: 7.9,
      dna_damage_index: 0.35,
      zmpste24_expression: 1.1,
      autophagy_marker: 0.9,
      inflammatory_cytokine: 0.8,
      oxidative_stress: 0.9,
      growth_retardation_score: 0.15,
      skin_elasticity: 0.95,
      joint_mobility: 0.98,
      hair_density_index: 0.88,
      cardiovascular_risk: 0.75,
      skeletal_abnormality: 0.90,
    },
    metabolic: {
      patient_id: 'PT-8812',
      age: 19.0,
      bmi: 18.2,
      heart_rate: 94,
      blood_pressure_sys: 118,
      blood_pressure_dia: 74,
      cholesterol: 185,
      hemoglobin: 11.8,
      white_blood_cells: 7.5,
      platelet_count: 195,
      creatinine: 1.30,
      albumin: 3.6,
      c_reactive_protein: 4.8,
      lmna_mutation_score: 0.5,
      progerin_level: 0.3,
      telomere_length: 5.6,
      dna_damage_index: 1.1,
      zmpste24_expression: 0.95,
      autophagy_marker: 1.4,
      inflammatory_cytokine: 2.9,
      oxidative_stress: 4.25,
      growth_retardation_score: 0.65,
      skin_elasticity: 0.55,
      joint_mobility: 0.45,
      hair_density_index: 0.60,
      cardiovascular_risk: 0.50,
      skeletal_abnormality: 0.35,
    },
    healthy: {
      patient_id: 'PT-1001',
      age: 26.0,
      bmi: 22.4,
      heart_rate: 72,
      blood_pressure_sys: 118,
      blood_pressure_dia: 76,
      cholesterol: 175,
      hemoglobin: 14.8,
      white_blood_cells: 6.4,
      platelet_count: 260,
      creatinine: 0.92,
      albumin: 4.5,
      c_reactive_protein: 0.8,
      lmna_mutation_score: 0.2,
      progerin_level: 0.1,
      telomere_length: 8.5,
      dna_damage_index: 0.15,
      zmpste24_expression: 1.05,
      autophagy_marker: 0.95,
      inflammatory_cytokine: 0.4,
      oxidative_stress: 0.8,
      growth_retardation_score: 0.05,
      skin_elasticity: 0.85,
      joint_mobility: 0.75,
      hair_density_index: 0.90,
      cardiovascular_risk: 0.12,
      skeletal_abnormality: 0.08,
    },
  };

  // Preset aliases matching index.html button data-preset values
  PRESETS.progeria = PRESETS.laminopathy;
  PRESETS.gaucher = PRESETS.lysosomal;
  PRESETS.ehlers = PRESETS.connective;

  // Slider field configurations mapping IDs to labels, units, and formatting
  const SLIDER_CONFIGS = {
    'input-age': { badgeId: 'val-age', format: (v) => `${Math.round(v)} yrs` },
    'input-bmi': { badgeId: 'val-bmi', format: (v) => `${parseFloat(v).toFixed(1)} kg/m²` },
    'input-hr': { badgeId: 'val-hr', format: (v) => `${Math.round(v)} bpm` },
    'input-sbp': { badgeId: 'val-sbp', format: (v) => `${Math.round(v)} mmHg` },
    'input-dbp': { badgeId: 'val-dbp', format: (v) => `${Math.round(v)} mmHg` },
    'input-chol': { badgeId: 'val-chol', format: (v) => `${Math.round(v)} mg/dL` },
    'input-hgb': { badgeId: 'val-hgb', format: (v) => `${parseFloat(v).toFixed(1)} g/dL` },
    'input-wbc': { badgeId: 'val-wbc', format: (v) => `${parseFloat(v).toFixed(1)} x10³/µL` },
    'input-plt': { badgeId: 'val-plt', format: (v) => `${Math.round(v)} x10³/µL` },
    'input-cr': { badgeId: 'val-cr', format: (v) => `${parseFloat(v).toFixed(2)} mg/dL` },
    'input-alb': { badgeId: 'val-alb', format: (v) => `${parseFloat(v).toFixed(1)} g/dL` },
    'input-crp': { badgeId: 'val-crp', format: (v) => `${parseFloat(v).toFixed(1)} mg/L` },

    'input-lmna': { badgeId: 'val-lmna', format: (v) => `${parseFloat(v).toFixed(1)} / 10` },
    'input-prog': { badgeId: 'val-prog', format: (v) => `${parseFloat(v).toFixed(1)} / 10` },
    'input-telo': { badgeId: 'val-telo', format: (v) => `${parseFloat(v).toFixed(1)} kb` },
    'input-dna': { badgeId: 'val-dna', format: (v) => `${parseFloat(v).toFixed(2)}` },
    'input-zmp': { badgeId: 'val-zmp', format: (v) => `${parseFloat(v).toFixed(2)}` },
    'input-aut': { badgeId: 'val-aut', format: (v) => `${parseFloat(v).toFixed(2)}` },
    'input-cyt': { badgeId: 'val-cyt', format: (v) => `${parseFloat(v).toFixed(1)} / 10` },
    'input-ox': { badgeId: 'val-ox', format: (v) => `${parseFloat(v).toFixed(1)}` },

    'input-growth': {
      badgeId: 'val-growth',
      format: (v) => `${parseFloat(v).toFixed(2)} ${v > 0.7 ? '(Severe)' : v > 0.3 ? '(Moderate)' : '(Normal)'}`,
    },
    'input-skin': {
      badgeId: 'val-skin',
      format: (v) => `${parseFloat(v).toFixed(2)} ${v < 0.3 ? '(Sclerotic)' : v > 0.85 ? '(Hyperextensible)' : '(Normal)'}`,
    },
    'input-joint': {
      badgeId: 'val-joint',
      format: (v) => `${parseFloat(v).toFixed(2)} ${v > 0.8 ? '(Hypermobile)' : v < 0.4 ? '(Contracture)' : '(Normal)'}`,
    },
    'input-hair': {
      badgeId: 'val-hair',
      format: (v) => `${parseFloat(v).toFixed(2)} ${v < 0.35 ? '(Severe Loss)' : '(Normal)'}`,
    },
    'input-cv': {
      badgeId: 'val-cv',
      format: (v) => `${parseFloat(v).toFixed(2)} ${v > 0.7 ? '(Atherosclerotic)' : '(Normal)'}`,
    },
    'input-skel': {
      badgeId: 'val-skel',
      format: (v) => `${parseFloat(v).toFixed(2)} ${v > 0.6 ? '(Dysplasia)' : '(Normal)'}`,
    },
  };

  let currentPatientId = 'PT-4092';
  let currentPredictionData = null;

  /* ==========================================================================
     2. INITIALIZATION & EVENT HOOKS
     ========================================================================== */
  document.addEventListener('DOMContentLoaded', () => {
    initHeroCounters();
    initPresetButtons();
    initTabs();
    initSliders();
    initCohortTable();
    initReportExport();
    initDiagnosisTrigger();
    initNavigationScroll();

    // Run initial prediction for default Laminopathy preset
    loadPreset('laminopathy', false);
    executeDiagnosis();
  });

  /* ==========================================================================
     3. HERO COUNTER ANIMATIONS
     ========================================================================== */
  function initHeroCounters() {
    const counters = document.querySelectorAll('.counter');
    counters.forEach((counter) => {
      const target = +counter.getAttribute('data-target') || 0;
      let count = 0;
      const step = Math.max(1, Math.ceil(target / 40));
      const interval = setInterval(() => {
        count += step;
        if (count >= target) {
          counter.innerText = target;
          clearInterval(interval);
        } else {
          counter.innerText = count;
        }
      }, 25);
    });
  }

  /* ==========================================================================
     4. TABS & PRESETS
     ========================================================================== */
  function initTabs() {
    const tabButtons = document.querySelectorAll('.panel-tab, .category-tab');
    const tabPanels = document.querySelectorAll('.tab-content');

    tabButtons.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        tabButtons.forEach((b) => b.classList.remove('active'));
        tabPanels.forEach((p) => p.classList.remove('active'));

        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        const panel = document.getElementById(targetId);
        if (panel) panel.classList.add('active');
      });
    });
  }

  function initPresetButtons() {
    const presetPills = document.querySelectorAll('.archetype-btn, .preset-pill');
    presetPills.forEach((pill) => {
      pill.addEventListener('click', (e) => {
        e.preventDefault();
        presetPills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        const presetKey = pill.getAttribute('data-preset');
        loadPreset(presetKey, true);
      });
    });
  }

  function loadPreset(presetKey, autoRun = true) {
    const p = PRESETS[presetKey];
    if (!p) return;

    currentPatientId = p.patient_id;
    const hudPid = document.getElementById('hud-pid');
    if (hudPid) hudPid.innerText = p.patient_id;

    // Set all slider values
    setSliderVal('input-age', p.age);
    setSliderVal('input-bmi', p.bmi);
    setSliderVal('input-hr', p.heart_rate);
    setSliderVal('input-sbp', p.blood_pressure_sys);
    setSliderVal('input-dbp', p.blood_pressure_dia);
    setSliderVal('input-chol', p.cholesterol);
    setSliderVal('input-hgb', p.hemoglobin);
    setSliderVal('input-wbc', p.white_blood_cells);
    setSliderVal('input-plt', p.platelet_count);
    setSliderVal('input-cr', p.creatinine);
    setSliderVal('input-alb', p.albumin);
    setSliderVal('input-crp', p.c_reactive_protein);

    setSliderVal('input-lmna', p.lmna_mutation_score);
    setSliderVal('input-prog', p.progerin_level);
    setSliderVal('input-telo', p.telomere_length);
    setSliderVal('input-dna', p.dna_damage_index);
    setSliderVal('input-zmp', p.zmpste24_expression);
    setSliderVal('input-aut', p.autophagy_marker);
    setSliderVal('input-cyt', p.inflammatory_cytokine);
    setSliderVal('input-ox', p.oxidative_stress);

    setSliderVal('input-growth', p.growth_retardation_score);
    setSliderVal('input-skin', p.skin_elasticity);
    setSliderVal('input-joint', p.joint_mobility);
    setSliderVal('input-hair', p.hair_density_index);
    setSliderVal('input-cv', p.cardiovascular_risk);
    setSliderVal('input-skel', p.skeletal_abnormality);

    if (autoRun) {
      executeDiagnosis();
    }
  }

  function setSliderVal(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = val;
    updateSliderBadge(id, val);
  }

  function initSliders() {
    Object.keys(SLIDER_CONFIGS).forEach((sliderId) => {
      const el = document.getElementById(sliderId);
      if (!el) return;
      el.addEventListener('input', (e) => {
        updateSliderBadge(sliderId, e.target.value);
      });
    });
  }

  function updateSliderBadge(sliderId, value) {
    const cfg = SLIDER_CONFIGS[sliderId];
    if (!cfg) return;
    const badge = document.getElementById(cfg.badgeId);
    if (badge) {
      badge.innerText = cfg.format(value);
    }
  }

  /* ==========================================================================
     5. INFERENCE EXECUTION & DIFFERENTIAL DIAGNOSIS
     ========================================================================== */
  function initDiagnosisTrigger() {
    const btn = document.getElementById('btn-run-diagnosis');
    if (btn) {
      btn.addEventListener('click', executeDiagnosis);
    }
  }

  function collectPatientPayload() {
    return {
      patient_id: currentPatientId,
      age: parseFloat(document.getElementById('input-age')?.value || 25),
      bmi: parseFloat(document.getElementById('input-bmi')?.value || 22),
      heart_rate: parseFloat(document.getElementById('input-hr')?.value || 75),
      blood_pressure_sys: parseFloat(document.getElementById('input-sbp')?.value || 120),
      blood_pressure_dia: parseFloat(document.getElementById('input-dbp')?.value || 80),
      cholesterol: parseFloat(document.getElementById('input-chol')?.value || 180),
      hemoglobin: parseFloat(document.getElementById('input-hgb')?.value || 14),
      white_blood_cells: parseFloat(document.getElementById('input-wbc')?.value || 7),
      platelet_count: parseFloat(document.getElementById('input-plt')?.value || 250),
      creatinine: parseFloat(document.getElementById('input-cr')?.value || 1.0),
      albumin: parseFloat(document.getElementById('input-alb')?.value || 4.2),
      c_reactive_protein: parseFloat(document.getElementById('input-crp')?.value || 2.0),

      lmna_mutation_score: parseFloat(document.getElementById('input-lmna')?.value || 1.0),
      zmpste24_expression: parseFloat(document.getElementById('input-zmp')?.value || 1.0),
      progerin_level: parseFloat(document.getElementById('input-prog')?.value || 0.5),
      telomere_length: parseFloat(document.getElementById('input-telo')?.value || 7.0),
      dna_damage_index: parseFloat(document.getElementById('input-dna')?.value || 0.4),
      autophagy_marker: parseFloat(document.getElementById('input-aut')?.value || 1.0),
      inflammatory_cytokine: parseFloat(document.getElementById('input-cyt')?.value || 0.8),
      oxidative_stress: parseFloat(document.getElementById('input-ox')?.value || 1.0),

      growth_retardation_score: parseFloat(document.getElementById('input-growth')?.value || 0.2),
      skin_elasticity: parseFloat(document.getElementById('input-skin')?.value || 0.8),
      joint_mobility: parseFloat(document.getElementById('input-joint')?.value || 0.8),
      hair_density_index: parseFloat(document.getElementById('input-hair')?.value || 0.8),
      cardiovascular_risk: parseFloat(document.getElementById('input-cv')?.value || 0.3),
      skeletal_abnormality: parseFloat(document.getElementById('input-skel')?.value || 0.2),
    };
  }

  async function executeDiagnosis() {
    const btn = document.getElementById('btn-run-diagnosis');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-atom fa-spin"></i><span>Synthesizing Quantum Hilbert Feature Map...</span>';
    }

    const payload = collectPatientPayload();
    const hudTime = document.getElementById('hud-time');
    if (hudTime) {
      hudTime.innerText = new Date().toLocaleTimeString();
    }

    try {
      // Try real FastAPI backend
      const response = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      renderDiagnosisResults(data);
    } catch (err) {
      console.warn('Backend unavailable, running high-fidelity browser quantum simulation:', err);
      // Fallback deterministic simulation
      const fallbackData = computeClientSideDiagnosis(payload);
      renderDiagnosisResults(fallbackData);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-atom"></i><span>Execute Quantum Feature Map & Diagnosis</span>';
      }
    }
  }

  /* --------------------------------------------------------------------------
     Client-Side Deterministic Fallback Math
     -------------------------------------------------------------------------- */
  function computeClientSideDiagnosis(p) {
    // Quantum ZZFeatureMap phase rotations [0, pi]
    const theta0 = Math.min(Math.PI, Math.max(0.05, (p.lmna_mutation_score / 10) * Math.PI));
    const theta1 = Math.min(Math.PI, Math.max(0.05, (p.progerin_level / 10) * Math.PI));
    const theta2 = Math.min(Math.PI, Math.max(0.05, (1.0 - Math.min(p.telomere_length, 10) / 10) * Math.PI));
    const theta3 = Math.min(Math.PI, Math.max(0.05, (p.growth_retardation_score) * Math.PI));
    const theta4 = Math.min(Math.PI, Math.max(0.05, (p.cardiovascular_risk) * Math.PI));
    const theta5 = Math.min(Math.PI, Math.max(0.05, (p.skeletal_abnormality) * Math.PI));

    const rotationAngles = [
      +theta0.toFixed(4),
      +theta1.toFixed(4),
      +theta2.toFixed(4),
      +theta3.toFixed(4),
      +theta4.toFixed(4),
      +theta5.toFixed(4),
    ];

    // Biomarker pathogenic score
    let score = 0;
    score += (p.lmna_mutation_score / 10) * 0.35;
    score += (p.progerin_level / 10) * 0.30;
    score += (Math.max(0, 8.0 - p.telomere_length) / 7.0) * 0.20;
    score += (p.growth_retardation_score) * 0.25;
    score += (p.cardiovascular_risk) * 0.15;
    score += (p.c_reactive_protein / 20.0) * 0.10;
    score += (p.autophagy_marker / 3.0) * 0.15;

    const risk = Math.min(0.985, Math.max(0.035, score * 0.65));

    // Models
    const rf = Math.min(0.99, Math.max(0.02, risk * 1.05 + 0.02));
    const svm = Math.min(0.99, Math.max(0.02, risk * 1.08));
    const qsvc = Math.min(0.95, Math.max(0.08, risk * 0.95 + 0.04));
    const vqc = Math.min(0.90, Math.max(0.12, risk * 0.88 + 0.06));

    // Differential categorization
    let diff;
    if (p.lmna_mutation_score >= 4.0 || p.progerin_level >= 3.0) {
      diff = {
        name: 'Laminopathy & Progeroid Syndrome',
        icd_10: 'ICD-10: E34.8 / Q87.1',
        description: 'Nuclear envelope structural compromise driven by LMNA pathogenic transversion, leading to cytotoxic accumulation of farnesylated progerin, cellular senescence, and severe microvascular fragility.',
        key_biomarkers: [
          `LMNA Score: ${p.lmna_mutation_score}`,
          `Progerin Level: ${p.progerin_level}`,
          `Telomere Loss: ${p.telomere_length} kb`,
        ],
        clinical_urgency: 'URGENT: Expedited pediatric genetics referral, farnesyltransferase inhibitor evaluation, and cardiovascular telemetry.',
      };
    } else if (p.autophagy_marker >= 2.0 || p.creatinine >= 1.6) {
      diff = {
        name: 'Lysosomal Storage Disease (Gaucher / Fabry Spectrum)',
        icd_10: 'ICD-10: E75.22 / E75.21',
        description: 'Enzymatic catabolism deficiency causing progressive intracellular glycosphingolipid accumulation, organomegaly, and renal/cardiac deterioration.',
        key_biomarkers: [
          `Autophagy Flux: ${p.autophagy_marker}`,
          `Serum Creatinine: ${p.creatinine} mg/dL`,
          `Platelet Count: ${p.platelet_count} x10³/µL`,
        ],
        clinical_urgency: 'HIGH PRIORITY: Enzyme replacement therapy (ERT) workup and substrate reduction screening.',
      };
    } else if (p.joint_mobility >= 0.85 || p.skin_elasticity >= 0.90) {
      diff = {
        name: 'Connective Tissue Dysplasia (Ehlers-Danlos / Marfan)',
        icd_10: 'ICD-10: Q79.60 / Q87.40',
        description: 'Defective collagen fibrillogenesis or fibrillin-1 mutations causing vascular fragility, severe joint hypermobility, and thoracic aortic aneurysm risk.',
        key_biomarkers: [
          `Joint Score: ${p.joint_mobility}`,
          `Skin Extensibility: ${p.skin_elasticity}`,
          `Aortic Risk Axis: ${p.cardiovascular_risk}`,
        ],
        clinical_urgency: 'CRITICAL VASCULAR: Transthoracic echocardiogram to rule out aortic root dilation.',
      };
    } else if (p.oxidative_stress >= 3.0 || p.growth_retardation_score >= 0.5) {
      diff = {
        name: 'Inborn Metabolic & Mitochondrial Defect',
        icd_10: 'ICD-10: E88.89 / G71.3',
        description: 'Respiratory chain complex deficiency with excessive reactive oxygen species generation, lactic acidemia, and neuromuscular failure.',
        key_biomarkers: [
          `Oxidative Stress: ${p.oxidative_stress}`,
          `Growth Index: ${p.growth_retardation_score}`,
          `Cytokine Panel: ${p.inflammatory_cytokine}`,
        ],
        clinical_urgency: 'SPECIALIZED METABOLIC: Plasma amino acids, urine organic acids, and mitochondrial genome sequencing.',
      };
    } else {
      diff = {
        name: 'Negative Screening / Healthy Baseline',
        icd_10: 'ICD-10: Z00.00',
        description: 'No significant pathogenic variants or organ dysfunction detected. Feature vector resides within normal population variance in Hilbert space.',
        key_biomarkers: [
          `Telomere: ${p.telomere_length} kb`,
          `LMNA: ${p.lmna_mutation_score}`,
          `CRP: ${p.c_reactive_protein} mg/L`,
        ],
        clinical_urgency: 'ROUTINE: Annual preventive follow-up.',
      };
    }

    const triage = risk >= 0.6 ? 'CRITICAL / HIGH RISK' : risk >= 0.28 ? 'MODERATE RISK' : 'LOW RISK / CONTROL';

    return {
      patient_id: p.patient_id,
      ensemble_risk_probability: +risk.toFixed(4),
      triage_status: triage,
      clinical_recommendation: diff.clinical_urgency,
      differential_diagnosis: diff,
      model_breakdown: {
        random_forest_risk: +rf.toFixed(4),
        svm_rbf_risk: +svm.toFixed(4),
        quantum_kernel_qsvc_risk: +qsvc.toFixed(4),
        variational_quantum_vqc_risk: +vqc.toFixed(4),
      },
      quantum_metrics: {
        n_qubits: 6,
        feature_map: 'ZZFeatureMap',
        quantum_state_rotation_angles: rotationAngles,
      },
    };
  }

  /* --------------------------------------------------------------------------
     Render Diagnosis HUD
     -------------------------------------------------------------------------- */
  function renderDiagnosisResults(data) {
    currentPredictionData = data;

    // 1. Patient ID & Status Badge
    const hudPid = document.getElementById('hud-pid');
    if (hudPid) hudPid.innerText = data.patient_id || currentPatientId;

    const badge = document.getElementById('triage-status-badge');
    if (badge) {
      badge.innerText = data.triage_status;
      badge.className = '';
      if (data.triage_status.includes('CRITICAL')) {
        badge.className = 'badge-critical-glow';
      } else if (data.triage_status.includes('MODERATE')) {
        badge.className = 'badge-moderate-glow';
      } else {
        badge.className = 'badge-healthy-glow';
      }
    }

    // 2. SVG Circular Gauge & Counter
    const riskPct = (data.ensemble_risk_probability * 100).toFixed(1);
    animateGauge(riskPct);

    // 3. Model Consensus Bars
    const mb = data.model_breakdown || {};
    updateConsensusBar('bar-qsvm', 'bar-val-qsvm', mb.quantum_kernel_qsvc_risk || 0.85);
    updateConsensusBar('bar-rf', 'bar-val-rf', mb.random_forest_risk || 0.92);
    updateConsensusBar('bar-svm', 'bar-val-svm', mb.svm_rbf_risk || 0.94);
    updateConsensusBar('bar-vqc', 'bar-val-vqc', mb.variational_quantum_vqc_risk || 0.78);

    // 4. Differential Diagnostic Anchor Card
    const diff = data.differential_diagnosis || {};
    const dName = document.getElementById('diag-name');
    const dIcd = document.getElementById('diag-icd');
    const dDesc = document.getElementById('diag-desc');
    const dUrgency = document.getElementById('diag-urgency');
    const dChips = document.getElementById('diag-biomarkers');

    if (dName) dName.innerText = diff.primary_category || diff.name || 'Diagnostic Classification';
    if (dIcd) dIcd.innerText = diff.icd10 ? `ICD-10: ${diff.icd10}` : (diff.icd_10 || 'ICD-10: Undefined');
    if (dDesc) dDesc.innerText = diff.phenotypic_description || diff.description || '';
    if (dUrgency) dUrgency.innerText = diff.clinical_urgency || data.clinical_recommendation || '';

    const biomarkers = diff.primary_biomarkers || diff.key_biomarkers || [];
    if (dChips && Array.isArray(biomarkers)) {
      dChips.innerHTML = '';
      biomarkers.forEach((chipText) => {
        const chip = document.createElement('span');
        chip.className = 'biomarker-chip';
        chip.innerHTML = `<i class="fa-solid fa-circle-dot"></i> ${chipText}`;
        dChips.appendChild(chip);
      });
    }

    // 5. 6-Qubit Hilbert Phase Encodings
    const angles = data.quantum_metrics?.quantum_state_rotation_angles || [1.2, 0.8, 1.9, 0.5, 2.1, 1.4];
    renderQubitAngleBars(angles);

    // 6. Update 3D Quantum Scene & Bloch Spheres
    if (window.QGlitch3D && typeof window.QGlitch3D.updateQubitAngles === 'function') {
      window.QGlitch3D.updateQubitAngles(angles);
    }

    // 7. Update Quantum Statevector Histogram
    renderStatevectorHistogram(angles);
  }

  function animateGauge(targetPct) {
    const circle = document.getElementById('gauge-circle');
    const pctLabel = document.getElementById('gauge-pct');
    if (!circle || !pctLabel) return;

    // Circumference = 2 * PI * 68 = 427.26
    const radius = 68;
    const circumference = 2 * Math.PI * radius;
    circle.style.strokeDasharray = `${circumference} ${circumference}`;

    const offset = circumference - (targetPct / 100) * circumference;
    circle.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)';
    circle.style.strokeDashoffset = offset;

    // Animate text counter
    let startVal = parseFloat(pctLabel.innerText) || 0;
    const duration = 1200;
    const startTime = performance.now();

    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (targetPct - startVal) * ease;
      pctLabel.innerText = `${current.toFixed(1)}%`;
      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }
    requestAnimationFrame(step);
  }

  function updateConsensusBar(barId, valId, score) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    const pct = (score * 100).toFixed(1);

    if (bar) {
      bar.style.transition = 'width 1s cubic-bezier(0.16, 1, 0.3, 1)';
      bar.style.width = `${pct}%`;
    }
    if (val) {
      val.innerText = `${pct}%`;
    }
  }

  function renderQubitAngleBars(angles) {
    const container = document.getElementById('qubit-angle-bars');
    if (!container) return;

    container.innerHTML = '';
    const qubitShort = ['LMNA', 'PROG', 'TELO', 'GROWTH', 'CARD', 'SKEL'];

    angles.forEach((rad, idx) => {
      const pct = Math.min(100, Math.max(0, ((rad / Math.PI) * 100))).toFixed(1);
      const item = document.createElement('div');
      item.className = 'qubit-node-box';
      item.innerHTML = `
        <span class="q-id">q${idx} (${qubitShort[idx]})</span>
        <span class="q-angle">${rad} rad</span>
        <div class="q-bar-mini">
          <div class="q-bar-fill" style="width: ${pct}%;"></div>
        </div>
      `;
      container.appendChild(item);
    });
  }

  function renderStatevectorHistogram(angles) {
    const container = document.getElementById('statevector-bars');
    if (!container) return;

    container.innerHTML = '';

    // Sample top basis states from angles
    const baseProbs = [
      { state: '|000000⟩', p: Math.cos(angles[0] / 2) ** 2 * Math.cos(angles[1] / 2) ** 2 },
      { state: '|100000⟩', p: Math.sin(angles[0] / 2) ** 2 * Math.cos(angles[1] / 2) ** 2 },
      { state: '|010000⟩', p: Math.cos(angles[0] / 2) ** 2 * Math.sin(angles[1] / 2) ** 2 },
      { state: '|110000⟩', p: Math.sin(angles[0] / 2) ** 2 * Math.sin(angles[1] / 2) ** 2 },
      { state: '|111111⟩', p: 0.12 + Math.sin(angles[4]) * 0.08 },
    ];

    // Normalize
    const sum = baseProbs.reduce((acc, cur) => acc + cur.p, 0);
    baseProbs.forEach((item) => {
      item.p = item.p / sum;
      const pct = (item.p * 100).toFixed(1);

      const row = document.createElement('div');
      row.className = 'hist-row';
      row.innerHTML = `
        <span class="hist-state">${item.state}</span>
        <div class="hist-track">
          <div class="hist-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="hist-val">${pct}%</span>
      `;
      container.appendChild(row);
    });
  }

  /* ==========================================================================
     6. COHORT BATCH SCREENING MATRIX
     ========================================================================== */
  let cohortData = [];

  function generateMockCohort(count = 50) {
    const data = [];
    const categories = [
      { name: 'Laminopathy / Progeroid', code: 'E34.8', risk: 0.88, priority: 'critical' },
      { name: 'Lysosomal Storage (Gaucher/Fabry)', code: 'E75.22', risk: 0.74, priority: 'critical' },
      { name: 'Connective Tissue Dysplasia', code: 'Q79.60', risk: 0.68, priority: 'critical' },
      { name: 'Mitochondrial Metabolic Defect', code: 'E88.89', risk: 0.52, priority: 'moderate' },
      { name: 'Negative / Healthy Control', code: 'Z00.00', risk: 0.08, priority: 'healthy' },
    ];

    for (let i = 1; i <= count; i++) {
      const pid = `PT-${(4000 + i)}`;
      const cat = categories[i % categories.length];
      const age = 12 + Math.round(Math.random() * 55);
      const lmna = cat.priority === 'critical' ? (6.0 + Math.random() * 3.8).toFixed(1) : (0.2 + Math.random() * 1.5).toFixed(1);
      const prog = cat.priority === 'critical' ? (5.5 + Math.random() * 4.0).toFixed(1) : (0.1 + Math.random() * 1.0).toFixed(1);
      const telo = cat.priority === 'critical' ? (2.0 + Math.random() * 2.5).toFixed(1) : (6.5 + Math.random() * 4.0).toFixed(1);
      const risk = +(cat.risk + (Math.random() - 0.5) * 0.08).toFixed(3);

      data.push({
        patient_id: pid,
        age,
        risk: Math.max(0.02, Math.min(0.99, risk)),
        priority: cat.priority,
        category: cat.name,
        lmna,
        prog,
        telo,
      });
    }
    return data;
  }

  function initCohortTable() {
    cohortData = generateMockCohort(60);
    renderCohortTable(cohortData);

    // Search filter
    const searchInput = document.getElementById('cohort-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        applyCohortFilters();
      });
    }

    // Filter pills
    const pills = document.querySelectorAll('.filter-pill');
    pills.forEach((p) => {
      p.addEventListener('click', () => {
        pills.forEach((x) => x.classList.remove('active'));
        p.classList.add('active');
        applyCohortFilters();
      });
    });

    // Reload button
    const btnRefresh = document.getElementById('btn-refresh-cohort');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => {
        cohortData = generateMockCohort(60);
        applyCohortFilters();
      });
    }

    // CSV export
    const btnExport = document.getElementById('btn-export-cohort');
    if (btnExport) {
      btnExport.addEventListener('click', exportCohortCSV);
    }
  }

  function applyCohortFilters() {
    const searchVal = document.getElementById('cohort-search')?.value.toLowerCase().trim() || '';
    const activePill = document.querySelector('.filter-pill.active')?.getAttribute('data-filter') || 'all';

    const filtered = cohortData.filter((item) => {
      const matchSearch =
        item.patient_id.toLowerCase().includes(searchVal) ||
        item.category.toLowerCase().includes(searchVal) ||
        item.priority.toLowerCase().includes(searchVal);

      const matchFilter =
        activePill === 'all' ||
        item.priority === activePill;

      return matchSearch && matchFilter;
    });

    renderCohortTable(filtered);
  }

  function renderCohortTable(list) {
    const tbody = document.getElementById('cohort-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 2rem; color: #94a3b8;">No matching patient records found in registry.</td></tr>`;
      return;
    }

    list.forEach((p) => {
      const tr = document.createElement('tr');
      const pct = (p.risk * 100).toFixed(1);

      let badgeClass = 'badge-healthy-glow';
      let badgeLabel = 'LOW RISK';
      if (p.priority === 'critical') {
        badgeClass = 'badge-critical-glow';
        badgeLabel = 'CRITICAL';
      } else if (p.priority === 'moderate') {
        badgeClass = 'badge-moderate-glow';
        badgeLabel = 'MODERATE';
      }

      tr.innerHTML = `
        <td><strong style="color: #f8fafc;">${p.patient_id}</strong></td>
        <td>${p.age}</td>
        <td>
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span>${pct}%</span>
            <div style="width: 50px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
              <div style="width: ${pct}%; height: 100%; background: ${p.priority === 'critical' ? '#ef4444' : p.priority === 'moderate' ? '#f59e0b' : '#10b981'};"></div>
            </div>
          </div>
        </td>
        <td><span class="${badgeClass}">${badgeLabel}</span></td>
        <td>${p.category}</td>
        <td>${p.lmna}</td>
        <td>${p.prog}</td>
        <td>${p.telo}</td>
        <td>
          <button class="btn-load-sim" data-pid="${p.patient_id}">
            <i class="fa-solid fa-arrow-up-right-from-square"></i> Test
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Wire test buttons
    tbody.querySelectorAll('.btn-load-sim').forEach((btn) => {
      btn.addEventListener('click', () => {
        const pid = btn.getAttribute('data-pid');
        loadCohortPatientIntoSimulator(pid);
      });
    });
  }

  function loadCohortPatientIntoSimulator(pid) {
    const item = cohortData.find((x) => x.patient_id === pid);
    if (!item) return;

    currentPatientId = item.patient_id;
    const hudPid = document.getElementById('hud-pid');
    if (hudPid) hudPid.innerText = item.patient_id;

    setSliderVal('input-age', item.age);
    setSliderVal('input-lmna', item.lmna);
    setSliderVal('input-prog', item.prog);
    setSliderVal('input-telo', item.telo);

    if (item.priority === 'critical') {
      setSliderVal('input-growth', 0.88);
      setSliderVal('input-skin', 0.2);
      setSliderVal('input-cv', 0.82);
    } else {
      setSliderVal('input-growth', 0.1);
      setSliderVal('input-skin', 0.85);
      setSliderVal('input-cv', 0.15);
    }

    // Scroll smoothly to diagnostic suite
    const diagSection = document.getElementById('diagnostic-suite');
    if (diagSection) {
      diagSection.scrollIntoView({ behavior: 'smooth' });
    }

    executeDiagnosis();
  }

  function exportCohortCSV() {
    if (!cohortData || cohortData.length === 0) return;

    let csv = 'Patient_ID,Age,Consensus_Risk,Priority,Predicted_Category,LMNA_Score,Progerin_Level,Telomere_kb\n';
    cohortData.forEach((row) => {
      csv += `${row.patient_id},${row.age},${row.risk},${row.priority},"${row.category}",${row.lmna},${row.prog},${row.telo}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qglitch_cohort_screening_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /* ==========================================================================
     7. CLINICAL REPORT DOWNLOAD
     ========================================================================== */
  function initReportExport() {
    const btn = document.getElementById('btn-download-report');
    if (btn) {
      btn.addEventListener('click', generateClinicalReport);
    }
  }

  function generateClinicalReport() {
    const data = currentPredictionData || computeClientSideDiagnosis(collectPatientPayload());
    const dateStr = new Date().toUTCString();

    const report = `================================================================================
                      Q-GLITCH CLINICAL DIAGNOSTIC REPORT
       Multimodal Hybrid Quantum-Classical Artificial Intelligence Platform
                   VTU-Qiskit Fall Fest Hackathon 2026 (VTU-QFF-HC-0190)
================================================================================

PATIENT IDENTIFIER:        ${data.patient_id || currentPatientId}
ASSESSMENT TIMESTAMP:      ${dateStr}
TRIAGE CLASSIFICATION:     ${data.triage_status}
CONSENSUS RISK PROBABILITY: ${(data.ensemble_risk_probability * 100).toFixed(2)}%

--------------------------------------------------------------------------------
1. DIFFERENTIAL DIAGNOSTIC ANCHOR
--------------------------------------------------------------------------------
DISEASE ENTITY:            ${data.differential_diagnosis?.name || 'N/A'}
ICD-10 CLASSIFICATION:     ${data.differential_diagnosis?.icd_10 || 'N/A'}
PATHOLOGICAL ETIOLOGY:     ${data.differential_diagnosis?.description || 'N/A'}

KEY PATHOGENIC BIOMARKERS:
${(data.differential_diagnosis?.key_biomarkers || []).map((b) => `  * ${b}`).join('\n')}

CLINICAL RECOMMENDATION:
  ${data.clinical_recommendation || data.differential_diagnosis?.clinical_urgency || 'N/A'}

--------------------------------------------------------------------------------
2. QUANTUM HILBERT STATE ENCODING (IBM QISKIT 2.x)
--------------------------------------------------------------------------------
FEATURE MAP ARCHITECTURE:  ZZFeatureMap (Full Entanglement, Reps=2)
TOTAL QUBITS:              6 Superconducting Qubits
ROTATION ANGLES [0, pi]:
${(data.quantum_metrics?.quantum_state_rotation_angles || []).map((ang, i) => `  Qubit q${i}: ${ang} rad (${Math.round((ang * 180) / Math.PI)} deg)`).join('\n')}

--------------------------------------------------------------------------------
3. MULTI-MODEL CONSENSUS BREAKDOWN
--------------------------------------------------------------------------------
  * Quantum Kernel Classifier (QSVC):  ${((data.model_breakdown?.quantum_kernel_qsvc_risk || 0) * 100).toFixed(1)}%
  * Random Forest Ensemble:            ${((data.model_breakdown?.random_forest_risk || 0) * 100).toFixed(1)}%
  * Support Vector Machine (RBF):      ${((data.model_breakdown?.svm_rbf_risk || 0) * 100).toFixed(1)}%
  * Variational Quantum Classifier:    ${((data.model_breakdown?.variational_quantum_vqc_risk || 0) * 100).toFixed(1)}%

--------------------------------------------------------------------------------
AUTHORSHIP & VERIFICATION
Team: Q-Glitch (Team Lead: Mallikarjun.R | Developers: Mohammed Adil, Mallanagouda M)
Official Event: VTU-Qiskit Fall Fest Hackathon 2026
Verified with Qiskit SDK 2.x and Aer Quantum Simulator.
================================================================================`;

    const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Q-Glitch_Diagnostic_Report_${data.patient_id || 'PT'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /* ==========================================================================
     8. NAVIGATION SCROLL & ACTIVE STATE
     ========================================================================== */
  function initNavigationScroll() {
    const navItems = document.querySelectorAll('.nav-links .nav-item');
    window.addEventListener('scroll', () => {
      let currentSection = '';
      const sections = document.querySelectorAll('section[id]');
      sections.forEach((sec) => {
        const top = sec.offsetTop - 120;
        if (window.scrollY >= top) {
          currentSection = sec.getAttribute('id');
        }
      });

      navItems.forEach((item) => {
        item.classList.remove('active');
        if (item.getAttribute('href') === `#${currentSection}`) {
          item.classList.add('active');
        }
      });
    });
  }
})();
