import React, { useEffect, useState } from "react";
import ClientCount from "./ClientCount";
import FilterOptionsModal from "./FilterOptionsModal";
import { useSearchParams } from "react-router-dom";
import TotalAmountReq from "./TotalAmountReq";
import TotalLoanStatus from "./TotalLoanStatus";
import { Spinner } from "@/components/ui/spinner";

function index() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [isLoading, setIsLoading] = useState(false)
  const [clientCount, setClientCount] = useState("");
  const [amountReq, setAmountReq] = useState("");
  const [totalLoanStatus, setTotalLoanStatus] = useState("");

  const [filters, setFilters] = useState({
    gender: searchParams.get("gender") || "all",
    married: searchParams.get("married") || "all",
  });

  useEffect(() => {
    const params = {};
    if (filters.gender != "all") params.gender = filters.gender;
    if (filters.married != "all") params.married = filters.married;
    setSearchParams(params);
  }, [filters]);

  const authFetch = (url) => {
    const token = localStorage.getItem("token");

    return fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${token}`
      }
    })
  }

  useEffect(() => {
    const fetchSummaryData = async () => {
      setIsLoading(true);

      const params = new URLSearchParams();
      if (filters.gender !== "all") params.append("gender", filters.gender);
      if (filters.married !== "all") params.append("married", filters.married);

      try {
        const [countRes, amountRes, loanRes] = await Promise.all([
          authFetch(`http://127.0.0.1:8000/api/clients/count?${params}`),
          authFetch(
            `http://127.0.0.1:8000/api/loan/total-amount-requested?${params}`
          ),
          authFetch(`http://127.0.0.1:8000/api/loan/total-loan-status?${params}`),
        ]);

        const [countData, amountData, loanData] = await Promise.all([
          countRes.json(),
          amountRes.json(),
          loanRes.json(),
        ]);

        setClientCount(countData);
        setAmountReq(amountData);
        setTotalLoanStatus(loanData);
      } catch (err) {
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSummaryData();
  }, [filters]);

  if (isLoading) {
    return (
      <div className="flex flex-col justify-center items-center md:w-[1250px] md:h-screen">
        <Spinner className="h-8 w-8 text-neutral-600" />
        <p className="text-neutral-600">Please Wait</p>
      </div>
    )
  }

  return (
    <div>
      <div className="md:pt-6">
        <FilterOptionsModal filters={filters} setFilters={setFilters} />
      </div>
      <div className="w-screen flex flex-col items-center gap-6 md:w-auto md:flex-row md:p-4">
        <div>
          <ClientCount clientCount={clientCount} />
        </div>
        <div>
          <TotalAmountReq amountReq={amountReq} />
        </div>
        <div>
          <TotalLoanStatus totalLoanStatus={totalLoanStatus} />
        </div>
      </div>
    </div>
  );
}

export default index;
