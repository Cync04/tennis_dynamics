const xSelect = document.getElementById('xColumn');
const filtersContainer = document.getElementById('filters');
const addFilterBtn = document.getElementById('addFilter');
const applyBtn = document.getElementById('applyBtn');
const chartImg = document.getElementById('chart');
const statusEl = document.getElementById('status');
const summaryEl = document.getElementById('summary');
const filterTemplate = document.getElementById('filterTemplate');

let columns = [];

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle('error', isError);
}

function makeOption(value, text) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = text;
  return option;
}

function addFilterRow() {
  const node = filterTemplate.content.firstElementChild.cloneNode(true);
  const columnSelect = node.querySelector('.filter-column');
  const removeButton = node.querySelector('.remove-filter');

  columns.forEach((col) => {
    columnSelect.appendChild(makeOption(col.name, col.name));
  });

  removeButton.addEventListener('click', () => {
    node.remove();
  });

  filtersContainer.appendChild(node);
}

function collectFilters() {
  const rows = Array.from(filtersContainer.querySelectorAll('.filter-row'));
  return rows
    .map((row) => ({
      column: row.querySelector('.filter-column').value,
      op: row.querySelector('.filter-op').value,
      value: row.querySelector('.filter-value').value,
    }))
    .filter((f) => f.column && f.op && f.value !== '');
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }
  return Number(value).toFixed(digits);
}

async function fetchMeta() {
  const response = await fetch('/api/meta');
  if (!response.ok) {
    throw new Error('Could not load metadata.');
  }
  return response.json();
}

async function renderPlot() {
  setStatus('Rendering chart...');

  const payload = {
    x_column: xSelect.value,
    filters: collectFilters(),
  };

  const response = await fetch('/api/plot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    chartImg.removeAttribute('src');
    summaryEl.innerHTML = '';
    setStatus(data.error || 'Failed to render chart.', true);
    return;
  }

  chartImg.src = `data:image/png;base64,${data.image}`;

  const s = data.summary;
  summaryEl.innerHTML = `
    <div>Rows used: <strong>${s.rows_used}</strong></div>
    <div>Won points: <strong>${s.won_points}</strong></div>
    <div>Lost points: <strong>${s.lost_points}</strong></div>
    <div>Won trend model: <strong>${s.won_trend_model || 'n/a'}</strong></div>
    <div>Lost trend model: <strong>${s.lost_trend_model || 'n/a'}</strong></div>
    <div>Won fit score (adj. R2): <strong>${formatNumber(s.won_fit_score)}</strong></div>
    <div>Lost fit score (adj. R2): <strong>${formatNumber(s.lost_fit_score)}</strong></div>
  `;

  setStatus('Chart updated.');
}

async function init() {
  try {
    const meta = await fetchMeta();
    columns = meta.columns;

    const numeric = columns.filter((c) => c.type === 'numeric');
    numeric.forEach((col) => {
      xSelect.appendChild(makeOption(col.name, col.name));
    });

    if (meta.default_x) {
      xSelect.value = meta.default_x;
    }

    addFilterRow();
    await renderPlot();
  } catch (err) {
    setStatus(err.message || 'Initialization failed.', true);
  }
}

addFilterBtn.addEventListener('click', addFilterRow);
applyBtn.addEventListener('click', renderPlot);

init();
