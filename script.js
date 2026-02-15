const datasetInfo = {
  india: {
    description:
      "Indian credit data shows strong linear separability. Logistic Regression performs consistently well due to stable repayment patterns.",
    roc: [0.78, 0.82],
    metrics: {
      roc: 0.78,
      ks: 0.42,
      recall: 0.71,
      precision: 0.69
    }
  },
  us: {
    description:
      "US credit data is highly non-linear and policy-sensitive. Random Forest outperforms linear models due to complex borrower behavior.",
    roc: [0.74, 0.86],
    metrics: {
      roc: 0.86,
      ks: 0.51,
      recall: 0.76,
      precision: 0.72
    }
  }
};

const select = document.getElementById("dataset-select");
const description = document.getElementById("dataset-description");

const rocValue = document.getElementById("rocValue");
const ksValue = document.getElementById("ksValue");
const recallValue = document.getElementById("recallValue");
const precisionValue = document.getElementById("precisionValue");

const ctx = document.getElementById("rocChart").getContext("2d");

let rocChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["Logistic Regression", "Random Forest"],
    datasets: [{
      data: datasetInfo.india.roc,
      backgroundColor: ["#60a5fa", "#4ade80"],
      borderRadius: 12,
      barThickness: 60
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true, max: 1 },
      x: { grid: { display: false } }
    }
  }
});

function updateDataset(key) {
  const data = datasetInfo[key];

  description.innerText = data.description;

  rocValue.innerText = data.metrics.roc;
  ksValue.innerText = data.metrics.ks;
  recallValue.innerText = data.metrics.recall;
  precisionValue.innerText = data.metrics.precision;

  rocChart.data.datasets[0].data = data.roc;
  rocChart.update();
}

select.addEventListener("change", e => updateDataset(e.target.value));

updateDataset("india");
