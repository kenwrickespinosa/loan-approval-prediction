import React, { useEffect, useState } from "react";
import ClientCount from "./ClientCount";
import FilterOptionsModal from "./FilterOptionsModal";
import { useSearchParams } from "react-router-dom";
import TotalAmountReq from "./TotalAmountReq";
import TotalLoanStatus from "./TotalLoanStatus";
import { Spinner } from "@/components/ui/spinner";
import ClientsWithLoans from "./ClientsWithLoans";

function index() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [isLoading, setIsLoading] = useState(false);
  const [clientCount, setClientCount] = useState("");
  const [amountReq, setAmountReq] = useState("");
  const [totalLoanStatus, setTotalLoanStatus] = useState("");
  const [clientsWithLoans, setClientsWithLoans] = useState([]);

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
        Authorization: `Bearer ${token}`,
      },
    });
  };

  useEffect(() => {
    const fetchSummaryData = async () => {
      setIsLoading(true);

      const params = new URLSearchParams();
      if (filters.gender !== "all") params.append("gender", filters.gender);
      if (filters.married !== "all") params.append("married", filters.married);

      try {
        const [countRes, amountRes, loanRes, clientsWithLoansRes] =
          await Promise.all([
            authFetch(`http://127.0.0.1:8000/api/clients/count?${params}`),
            authFetch(
              `http://127.0.0.1:8000/api/loan/total-amount-requested?${params}`
            ),
            authFetch(
              `http://127.0.0.1:8000/api/loan/total-loan-status?${params}`
            ),
            authFetch(
              `http://127.0.0.1:8000/api/clients/clients-with-and-without-loans?${params}`
            ),
          ]);

        const [countData, amountData, loanData, clientsLoansData] =
          await Promise.all([
            countRes.json(),
            amountRes.json(),
            loanRes.json(),
            clientsWithLoansRes.json(),
          ]);

        setClientCount(countData);
        setAmountReq(amountData);
        setTotalLoanStatus(loanData);
        setClientsWithLoans(clientsLoansData);
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
      <div className="flex flex-col justify-center items-center w-screen h-screen md:w-[1250px] md:h-screen">
        <Spinner className="h-8 w-8 text-neutral-600" />
        <p className="text-neutral-600">Please Wait</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <div>
        <FilterOptionsModal filters={filters} setFilters={setFilters} />
      </div>

      <div className="flex flex-col pt-4">
        <div className="md:flex md:gap-16">
          <div className="md:flex md:flex-col md:gap-4">
            <ClientCount clientCount={clientCount} />
            <TotalAmountReq amountReq={amountReq} />
          </div>
          <div>
            <TotalLoanStatus totalLoanStatus={totalLoanStatus} />
          </div>
        </div>

        <div>
          <ClientsWithLoans loans={clientsWithLoans} />
        </div>
      </div>
    </div>
  );
}

export default index;
