import { Card, CardContent, CardHeader } from "@/components/ui/card";
import React, { useEffect, useState } from "react";
import { FaRegUser } from "react-icons/fa";

function ClientCount({ filters }) {
  const [count, setCount] = useState("");

  useEffect(() => {
    const getCount = async () => {
      const params = new URLSearchParams(filters).toString();
      const token = localStorage.getItem("token");

      try {
        const res = await fetch(`http://127.0.0.1:8000/api/clients/count?${params}`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.message || "Failed to get client count");
        }

        setCount(data.total);
      } catch (err) {
        console.error(err);
      }
    };

    getCount();
  }, [filters]);

  return (
    <div>
      <Card className="border-l-8 border-l-blue-400 rounded-sm">
        <CardContent className="pb-0 p-5 flex items-center gap-10 bg-neutral-50">
          <div>
            <FaRegUser className="size-8" />
          </div>
          <div className="flex flex-col items-end gap-2">
            <p className="font-bold text-3xl">{count}</p>
            <p>Number of Applicants</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ClientCount;
