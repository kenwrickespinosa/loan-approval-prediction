import { Card, CardContent } from "@/components/ui/card";
import React, { useEffect, useState } from "react";

function TotalAmountReq({ amountReq }) {
  return (
    <Card className="border-l-8 mx-5 border-l-blue-400 rounded-sm md:mx-0">
      <CardContent className="pb-0 py-5 bg-neutral-50 md:w-[300px]">
        <div className="flex flex-col items-end gap-2">
          <p className="font-bold text-3xl">{amountReq.data}</p>
          <p>Total Loan Amount Requested</p>
        </div>
      </CardContent>
    </Card>
  );
}

export default TotalAmountReq;
