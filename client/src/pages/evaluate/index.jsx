import React, { useState } from "react";
import PredictionForm from "./PredictionForm";
import PredictionResult from "./PredictionResult";

function index() {
  const [predResult, setPredResult] = useState(null);

  return (
    <div className="min-h-screen flex justify-center items-center">
      <div className="w-[360px] md:flex md:ml-36 gap-6">
        <div className="p-6">
          <PredictionForm setPredResult={setPredResult} />
        </div>
        <div className="p-6">
          <span>Pred Result:</span>
          <PredictionResult predResult={predResult} />
        </div>
      </div>
    </div>
  );
}

export default index;
