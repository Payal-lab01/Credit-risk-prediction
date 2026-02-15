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
        backgroundColor: ["#3b82f6", "#22c55e"]
      }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: "#e5e7eb"
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          color: "#e5e7eb"
        }
      },
      x: {
        ticks: {
          color: "#e5e7eb"
        }
      }
    }
  }
});

function updateChart(dataset) {
  rocChart.data.datasets[0].data = datasetInfo[dataset].roc;
  rocChart.update();
}

/* ================= INITIAL LOAD ================= */

updateDataset("india");
