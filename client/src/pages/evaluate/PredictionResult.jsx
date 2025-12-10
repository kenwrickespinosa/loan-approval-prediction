import React from "react";

function PredictionResult({ predResult }) {
  if (!predResult) return null;  // predResult.prediction and predResult.approval_chance

  return (
    <div className="mt-4 p-4 border rounded-md md:w-[360px]">
      {predResult.prediction === 1 ? (
        <span className="text-green-600 font-bold">
          ✔ Loan Approved — Chance: {predResult.approval_chance}%
        </span>
      ) : (
        <span className="text-red-600 font-bold">
          ✖ Loan Not Approved — Chance: {predResult.approval_chance}%
        </span>
      )}
    </div>
  );
}

export default PredictionResult;
