import VertBarChart from "@/components/charts/VertBarChart";
import React, { useEffect, useState } from "react";

function TotalLoanStatus({ totalLoanStatus }) {
  // const [loanStatus, setLoanStatus] = useState({ approved: 0, rejected: 0 });

  // useEffect(() => {
  //   const getLoanStatus = async () => {
  //     const params = new URLSearchParams(filters).toString();
  //     const token = localStorage.getItem("token");

  //     try {
  //       const res = await fetch(`http://127.0.0.1:8000/api/loan/total-loan-status?${params}`, {
  //         method: "GET",
  //         headers: {
  //           "Content-Type": "application/json",
  //           Accept: "application/json",
  //           Authorization: `Bearer ${token}`,
  //         },
  //       });

  //       const data = await res.json();

  //       if (!res.ok) {
  //         throw new Error(data.message || "Failed to get loan status");
  //       }

  //       setLoanStatus(data);
  //     } catch (err) {
  //       console.error(err);
  //     }
  //   };

  //   getLoanStatus();
  // }, [filters]);

  return (
    <div>
      <VertBarChart loanStatus={totalLoanStatus} />
    </div>
  );
}

export default TotalLoanStatus;
