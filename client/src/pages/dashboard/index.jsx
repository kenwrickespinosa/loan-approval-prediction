import React, { useEffect, useState } from "react";
import ClientCount from "./ClientCount";
import FilterOptionsModal from "./FilterOptionsModal";
import { useSearchParams } from "react-router-dom";
import TotalAmountReq from "./TotalAmountReq";

function index() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [filters, setFilters] = useState({
    gender: searchParams.get("gender") || "all",
    married: searchParams.get("married") || "all",
  });

  useEffect(() => {
    const params = {};
    if (filters.gender != "all") params.gender = filters.gender;
    if (filters.married != "all") params.married = filters.married;
    setSearchParams(params);
  }, [filters])

  return (
    <div>
      <div>
        <FilterOptionsModal filters={filters} setFilters={setFilters} />
      </div>
      <div className="w-screen flex flex-col items-center gap-6 md:w-auto md:flex-row md:p-4">
        <div>
          <ClientCount filters={filters} />
        </div>
        <div>
          <TotalAmountReq filters={filters} />
        </div>
      </div>
    </div>
  );
}

export default index;
