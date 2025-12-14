import React, { useState } from "react";
import PredictionForm from "./PredictionForm";
import PredictionResult from "./PredictionResult";

function index() {
  const [predResult, setPredResult] = useState(null);
  const [selectedClient, setSelectedClient] = useState(null);

  const [isSaveEvaluation, setIsSaveEvaluation] = useState(null);

  const saveEvaluation = async (predResult, selectedClient) => {
    const token = localStorage.getItem("token");

    const payload = {
      client_id: selectedClient?.id || null,
      ...predResult.payload,
      loan_status: predResult.prediction === 1 ? "Y" : "N",
    };

    console.log("Saving Loan:", payload);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/loan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("failed");

      setIsSaveEvaluation("succeed");
    } catch (err) {
      console.error(err);
      setIsSaveEvaluation("failed");
    }
  };

  return (
    <div className="min-h-screen flex justify-center items-center">
      <div className="w-[360px] md:flex md:ml-36 gap-6">
        <div className="p-6">
          <PredictionForm
            setPredResult={setPredResult}
            setSelectedClient={setSelectedClient}
          />
        </div>
        <div className="p-6">
          {predResult ? (
            <PredictionResult
              predResult={predResult}
              selectedClient={selectedClient}
              onSaveEvaluation={saveEvaluation}
              saveStatus={isSaveEvaluation}
            />
          ) : (
            <p className="text-center bg-neutral-200 rounded-md py-2 md:w-[300px]">
              Waiting for evaluation...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default index;
