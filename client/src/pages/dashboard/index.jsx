import React, { useState } from "react";
import ClientCount from "./ClientCount";

function Dashboard() {
  const [filters, setFilters] = useState(null);
  
  return (
    <div>
      <div className="w-screen flex flex-col items-center gap-6 md:w-auto md:flex-row md:p-4">
        <div>
          <ClientCount />
        </div>
        <div>
          <ClientCount />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
