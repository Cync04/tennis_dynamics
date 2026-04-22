const xSelect = document.getElementById('xColumn');
const yMetricSelect = document.getElementById('yMetric');
const yMetricNote = document.getElementById('yMetricNote');
const yMetricBadge = document.getElementById('yMetricBadge');
const filtersContainer = document.getElementById('filters');
const addFilterBtn = document.getElementById('addFilter');
const applyBtn = document.getElementById('applyBtn');
const chartImg = document.getElementById('chart');
const statusEl = document.getElementById('status');
const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');
const summaryEl = document.getElementById('summary');
const filterTemplate = document.getElementById('filterTemplate');

let columns = [];
let columnMap = {};
let yMetrics = [];
let yMetricMap = {};
let progressTimer = null;
let isRendering = false;

// Operators offered when a filter column is numeric.
const NUMERIC_OPS = [
  { value: 'eq', label: '=' },
  { value: 'neq', label: '!=' },
  { value: 'gt', label: '>' },
  { value: 'gte', label: '>=' },
  { value: 'lt', label: '<' },
  { value: 'lte', label: '<=' },
  { value: 'between', label: 'between' },
];

// Operators offered when a filter column is categorical/text.
const CATEGORICAL_OPS = [
  { value: 'eq', label: '=' },
  { value: 'neq', label: '!=' },
  { value: 'contains', label: 'contains' },
  { value: 'in', label: 'in' },
];

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle('error', isError);
}

function startProgress() {
  let pct = 8;
  progressBar.style.width = `${pct}%`;
  progressWrap.classList.remove('hidden');

  if (progressTimer) {
    clearInterval(progressTimer);
  }

  // Indeterminate-style progress: climbs toward 90% while waiting for server.
  progressTimer = setInterval(() => {
    pct = Math.min(90, pct + Math.max(1, Math.round((100 - pct) * 0.07)));
    progressBar.style.width = `${pct}%`;
  }, 220);
}

function finishProgress(success) {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }

  progressBar.style.width = success ? '100%' : '0%';

  setTimeout(() => {
    if (success) {
      progressBar.style.width = '0%';
    }
    progressWrap.classList.add('hidden');
  }, success ? 220 : 0);
}

function makeOption(value, text) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = text;
  return option;
}

// Prefer win percentage when it is valid for the selected x-axis.
function getRecommendedMetric(compatibleMetrics) {
  const winPct = compatibleMetrics.find((metric) => metric.id === 'win_pct');
  if (winPct) {
    return winPct;
  }
  return compatibleMetrics[0] || null;
}

// Rebuild y-axis options whenever x-axis changes.
function updateYMetricOptions() {
  const selectedX = xSelect.value;
  const previous = yMetricSelect.value;

  yMetricSelect.innerHTML = '';
  const compatibleMetrics = yMetrics.filter((metric) => metric.compatible_x.includes(selectedX));
  const recommendedMetric = getRecommendedMetric(compatibleMetrics);

  compatibleMetrics.forEach((metric) => {
    const isRecommended = recommendedMetric && metric.id === recommendedMetric.id;
    const label = isRecommended ? `${metric.label} (recommended)` : metric.label;
    yMetricSelect.appendChild(makeOption(metric.id, label));
  });

  if (compatibleMetrics.some((m) => m.id === previous)) {
    yMetricSelect.value = previous;
  }

  if (!yMetricSelect.value && compatibleMetrics.length > 0) {
    yMetricSelect.value = recommendedMetric ? recommendedMetric.id : compatibleMetrics[0].id;
  }

  if (recommendedMetric) {
    yMetricBadge.textContent = `Recommended: ${recommendedMetric.label}`;
    yMetricBadge.classList.remove('hidden');
  } else {
    yMetricBadge.textContent = '';
    yMetricBadge.classList.add('hidden');
  }

  const selectedMetric = yMetricMap[yMetricSelect.value];
  yMetricNote.textContent = selectedMetric ? selectedMetric.description : '';
}

function resetSelectOptions(selectEl, options) {
  selectEl.innerHTML = '';
  options.forEach((op) => {
    selectEl.appendChild(makeOption(op.value, op.label));
  });
}

// Switch each filter row between numeric and categorical modes.
function updateFilterRowControls(row) {
  const columnSelect = row.querySelector('.filter-column');
  const opSelect = row.querySelector('.filter-op');
  const valueInput = row.querySelector('.filter-value');
  const valueSelect = row.querySelector('.filter-value-select');
  const col = columnMap[columnSelect.value];

  const previousOp = opSelect.value;
  if (!col || col.type === 'numeric') {
    resetSelectOptions(opSelect, NUMERIC_OPS);
    if (NUMERIC_OPS.some((op) => op.value === previousOp)) {
      opSelect.value = previousOp;
    }

    valueSelect.classList.add('hidden');
    valueInput.classList.remove('hidden');
    valueInput.placeholder = opSelect.value === 'between' ? 'min,max' : 'value';
    return;
  }

  resetSelectOptions(opSelect, CATEGORICAL_OPS);
  if (CATEGORICAL_OPS.some((op) => op.value === previousOp)) {
    opSelect.value = previousOp;
  }

  const shouldUseText = opSelect.value === 'contains' || opSelect.value === 'in' || !col.values || col.values.length === 0;
  if (shouldUseText) {
    valueSelect.classList.add('hidden');
    valueInput.classList.remove('hidden');
    valueInput.placeholder = opSelect.value === 'in' ? 'value1,value2' : 'value';
    return;
  }

  valueSelect.innerHTML = '';
  col.values.forEach((value) => {
    valueSelect.appendChild(makeOption(value, value));
  });

  valueInput.classList.add('hidden');
  valueSelect.classList.remove('hidden');
}

// Add one filter row and wire row-level event handlers.
function addFilterRow() {
  const node = filterTemplate.content.firstElementChild.cloneNode(true);
  const columnSelect = node.querySelector('.filter-column');
  const opSelect = node.querySelector('.filter-op');
  const removeButton = node.querySelector('.remove-filter');

  columns.forEach((col) => {
    columnSelect.appendChild(makeOption(col.name, col.name));
  });

  columnSelect.addEventListener('change', () => {
    updateFilterRowControls(node);
  });

  opSelect.addEventListener('change', () => {
    updateFilterRowControls(node);
  });

  removeButton.addEventListener('click', () => {
    node.remove();
  });

  filtersContainer.appendChild(node);
  updateFilterRowControls(node);
}

// Collect only complete filter rows to send to backend.
function collectFilters() {
  const rows = Array.from(filtersContainer.querySelectorAll('.filter-row'));
  return rows
    .map((row) => {
      const valueInput = row.querySelector('.filter-value');
      const valueSelect = row.querySelector('.filter-value-select');
      const useSelect = !valueSelect.classList.contains('hidden');

      return {
        column: row.querySelector('.filter-column').value,
        op: row.querySelector('.filter-op').value,
        value: useSelect ? valueSelect.value : valueInput.value,
      };
    })
    .filter((f) => f.column && f.op && f.value !== '');
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }
  return Number(value).toFixed(digits);
}

// Initial metadata load for axes, filter types, and compatibility maps.
async function fetchMeta() {
  const response = await fetch('/api/meta');
  if (!response.ok) {
    throw new Error('Could not load metadata.');
  }
  return response.json();
}

// Request new chart image + summary from backend.
async function renderPlot() {
  if (isRendering) {
    return;
  }

  isRendering = true;
  applyBtn.disabled = true;
  setStatus('Rendering chart...');
  startProgress();

  const payload = {
    x_column: xSelect.value,
    y_metric: yMetricSelect.value,
    filters: collectFilters(),
  };

  // Retry once for transient network/timeouts before showing failure.
  const maxAttempts = 2;
  let lastError = null;

  try {
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      try {
        const response = await fetch('/api/plot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        const contentType = response.headers.get('content-type') || '';
        const data = contentType.includes('application/json')
          ? await response.json()
          : { error: await response.text() };

        if (!response.ok) {
          // Server validation errors should not be retried.
          chartImg.removeAttribute('src');
          summaryEl.innerHTML = '';
          setStatus(data.error || 'Failed to render chart.', true);
          finishProgress(false);
          return;
        }

        chartImg.src = `data:image/png;base64,${data.image}`;

        const s = data.summary;
        summaryEl.innerHTML = `
          <div>Rows used: <strong>${s.rows_used}</strong></div>
          <div>Won points: <strong>${s.won_points}</strong></div>
          <div>Lost points: <strong>${s.lost_points}</strong></div>
          <div>Overall win chance: <strong>${formatNumber(s.overall_win_pct, 2)}%</strong></div>
          <div>Y metric: <strong>${s.selected_y_label || s.selected_y_metric}</strong></div>
          <div>Trend model: <strong>${s.trend_model || 'n/a'}</strong></div>
          <div>Fit score (adj. R2): <strong>${formatNumber(s.fit_score)}</strong></div>
        `;

        setStatus('Chart updated.');
        finishProgress(true);
        return;
      } catch (err) {
        lastError = err;
        if (attempt < maxAttempts) {
          setStatus(`Render attempt ${attempt} failed, retrying...`);
          continue;
        }
      } finally {
        clearTimeout(timeoutId);
      }
    }

    chartImg.removeAttribute('src');
    summaryEl.innerHTML = '';
    if (lastError && lastError.name === 'AbortError') {
      setStatus('Chart request timed out after 30s on both attempts. Try adding filters or retrying.', true);
    } else {
      const details = lastError && lastError.message ? ` (${lastError.message})` : '';
      setStatus(`Chart request failed after retry.${details}`, true);
    }
    finishProgress(false);
  } finally {
    isRendering = false;
    applyBtn.disabled = false;
  }
}

// App bootstrap sequence.
async function init() {
  try {
    const meta = await fetchMeta();
    columns = meta.columns;
    columnMap = Object.fromEntries(columns.map((col) => [col.name, col]));
    yMetrics = meta.y_metrics || [];
    yMetricMap = Object.fromEntries(yMetrics.map((metric) => [metric.id, metric]));

    const numeric = columns.filter((c) => c.type === 'numeric');
    numeric.forEach((col) => {
      xSelect.appendChild(makeOption(col.name, col.name));
    });

    if (meta.default_x) {
      xSelect.value = meta.default_x;
    }

    updateYMetricOptions();
    if (meta.default_y && yMetricMap[meta.default_y] && yMetricMap[meta.default_y].compatible_x.includes(xSelect.value)) {
      yMetricSelect.value = meta.default_y;
      yMetricNote.textContent = yMetricMap[meta.default_y].description;
    }

    addFilterRow();
    await renderPlot();
  } catch (err) {
    setStatus(err.message || 'Initialization failed.', true);
  }
}

addFilterBtn.addEventListener('click', addFilterRow);
// Keep y-axis options in sync with x-axis compatibility rules.
xSelect.addEventListener('change', () => {
  updateYMetricOptions();
});
// Update helper text when user selects a different y metric.
yMetricSelect.addEventListener('change', () => {
  const selectedMetric = yMetricMap[yMetricSelect.value];
  yMetricNote.textContent = selectedMetric ? selectedMetric.description : '';
});
applyBtn.addEventListener('click', renderPlot);

init();
