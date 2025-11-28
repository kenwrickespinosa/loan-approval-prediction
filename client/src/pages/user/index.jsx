import React, { useEffect, useState } from "react";
import CreateUserModal from "./CreateUserModal";
import { Input } from "@/components/ui/input";
import UsersList from "./UsersList";

function index() {
  const [clients, setClients] = useState([]);

  useEffect(() => {
    const fetchClients = async () => {
      try {
        const token = localStorage.getItem("token");

        const response = await fetch("http://127.0.0.1:8000/api/clients", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.message || "Failed to fetch clients");
        }

        setClients(data);
      } catch (error) {
        console.log(error);
      }
    };

    fetchClients();
  }, []);

  return (
    <div>
      <div className="flex">
        <Input className="md:w-2xs" />
        <CreateUserModal />
      </div>
      <div>
        <UsersList clients={clients} />
      </div>
    </div>
  );
}

export default index;
