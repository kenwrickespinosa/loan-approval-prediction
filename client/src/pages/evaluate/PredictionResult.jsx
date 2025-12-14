import React from "react";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";
import { Button } from "@/components/ui/button";

ChartJS.register(ArcElement, Tooltip, Legend);

function PredictionResult({ predResult, selectedClient, onSaveEvaluation, saveStatus }) {
  // predResult.prediction and predResult.approval_chance
  if (
    !predResult ||
    predResult.prediction == null ||
    predResult.approval_chance == null
  ) {
    return null;
  }

  const inputs = predResult.payload;

  const approval = predResult.approval_chance;
  const rejection = 100 - approval;

  const data = {
    labels: ["Approval", "Rejection"],
    datasets: [
      {
        data: [approval, rejection],
        backgroundColor: ["#60a5fa", "#e5e7eb"],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    cutout: "70%",
    plugins: {
      legend: { display: false },
    },
  };

  return (
    <div className="mt-4 p-4 border rounded-md md:w-[600px] md:mt-0 md:flex md:flex-col md:items-center">
      <p className="text-center mb-6">Evaluation Result</p>
      <div>
        <Doughnut data={data} options={options} />
      </div>
      <div className="flex flex-col gap-4 mt-6 md:flex-row md:gap-12">
        <div className="flex flex-col items-center py-4 gap-2 bg-neutral-100 rounded-md md:w-[200px]">
          <span className="text-neutral-600">Approval Chance</span>
          <span>{approval}</span>
        </div>
        <div className="flex flex-col items-center py-4 gap-2 bg-neutral-100 rounded-md md:w-[200px]">
          <span className="text-neutral-600">Result</span>
          <span className="text-center">
            {predResult.prediction === 1
              ? "Accept Loan"
              : "Not Recommended to Accept Loan"}
          </span>
        </div>
      </div>
      <Button
        type="submit"
        disabled={saveStatus === "loading"}
        onClick={() => onSaveEvaluation(predResult, selectedClient)}
        className="cursor-pointer max-md:w-full mt-6"
      >
        Save Evaluation
      </Button>
    </div>
  );
}

export default PredictionResult;
