/* ================= DATASET INTERACTION ================= */

const datasetInfo = {
  india: {
    description:
      "Indian credit data shows strong linear separability. Logistic Regression performs consistently well due to stable repayment patterns.",
    roc: [0.78, 0.82]
  },
  us: {
    description:
      "US credit data is highly non-linear and policy-sensitive. Random Forest outperforms linear models due to complex borrower behavior.",
    roc: [0.74, 0.86]
  }
};

const select = document.getElementById("dataset-select");
const description = document.getElementById("dataset-description");

function updateDataset(dataset) {
  description.innerText = datasetInfo[dataset].description;
  updateChart(dataset);
}

select.addEventListener("change", (e) => {
  updateDataset(e.target.value);
});

/* ================= CHART ================= */

const ctx = document.getElementById("rocChart").getContext("2d");

let rocChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["Logistic Regression", "Random Forest"],
    datasets: [
      {
        label: "ROC-AUC Score",
        data: datasetInfo.india.roc,
        backgroundColor: ["#60a5fa", "#4ade80"],
borderRadius: 12,
barThickness: 60

      }
    ]
  },
options: {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    },
    tooltip: {
      backgroundColor: "#020617",
      titleColor: "#e5e7eb",
      bodyColor: "#e5e7eb",
      borderColor: "#3b82f6",
      borderWidth: 1
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      max: 1,
      grid: {
        color: "rgba(148,163,184,0.15)"
      },
      ticks: {
        color: "#e5e7eb"
      }
    },
    x: {
      grid: {
        display: false
      },
      ticks: {
        color: "#e5e7eb"
      }
    }
  }
}
);

function updateChart(dataset) {
  rocChart.data.datasets[0].data = datasetInfo[dataset].roc;
  rocChart.update();
}

/* ================= INITIAL LOAD ================= */

updateDataset("india");
