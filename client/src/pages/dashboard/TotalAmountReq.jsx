import { Card, CardContent } from "@/components/ui/card";
import React, { useEffect, useState } from "react";

function TotalAmountReq({ amountReq }) {
  // const [total, setTotal] = useState("");

  // useEffect(() => {
  //   const getTotal = async () => {
  //     const params = new URLSearchParams(filters).toString();
  //     const token = localStorage.getItem("token");

  //     try {
  //       const res = await fetch(
  //         `http://127.0.0.1:8000/api/loan/total-amount-requested?${params}`,
  //         {
  //           method: "GET",
  //           headers: {
  //             "Content-Type": "application/json",
  //             Accept: "application/json",
  //             Authorization: `Bearer ${token}`,
  //           },
  //         }
  //       );

  //       const data = await res.json();

  //       if (!res.ok) {
  //         throw new Error(
  //           data.message || "Failed to retrive total loan amount requested"
  //         );
  //       } else {
  //         setTotal(data);
  //       }
  //     } catch (err) {
  //       console.error(err);
  //     }
  //   };

  //   getTotal();
  // }, [filters]);

  return (
    <div>
      <Card className="border-l-8 border-l-blue-400 rounded-sm">
        <CardContent className="pb-0 py-5 text-center bg-neutral-50">
          <div className="flex flex-col items-end gap-2">
            <p className="font-bold text-3xl">{amountReq.data}</p>
            <p>Total Loan Amount Requested</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default TotalAmountReq;
