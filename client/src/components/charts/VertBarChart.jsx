import {
  Chart as ChartJS,
  Legend,
  Tooltip,
  BarElement,
  LinearScale,
  CategoryScale,
} from "chart.js";
import React from "react";
import { Bar } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

function VertBarChart({ loanStatus }) {
  const labels = ["Accepted", "Rejected"];

  const data = {
    labels: ["Loans"],
    datasets: [
      {
        label: "Approved",
        data: [loanStatus.approved],
        backgroundColor: "#BEE3F8",
      },
      {
        label: "Rejected",
        data: [loanStatus.rejected],
        backgroundColor: "#FECACA",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animations: false,
    scales: {
      y: {
        begintAtZero: true,
        max:(loanStatus.approved + loanStatus.rejected + 2),
        ticks: {
          stepSize: 1,
          precision: 0,
        },
        title: {
          display: true,
          text: "Number of Loans",
        },
      },
      x: {
        title: {
          display: true,
        //   text: "Loan Status",
        },
      },
    },
  };

  return (
    <div className="w-full max-w-xl h-[300px]">
      <Bar
        data={data}
        options={options}
        key={`${loanStatus.approved}-${loanStatus.rejected}`}
      />
    </div>
  );
}

export default VertBarChart;
