import React, { useEffect, useState } from "react";
import CreateUserModal from "./CreateUserModal";
import { Input } from "@/components/ui/input";
import UsersList from "./UsersList";
import { Spinner } from "@/components/ui/spinner";

function index() {
  const [clients, setClients] = useState([]);

  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchClients = async () => {
      setIsLoading(true);

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
      } finally {
        setIsLoading(false);
      }
    };

    fetchClients();
  }, []);

  if (isLoading) {
    return (
      <div className="flex flex-col justify-center items-center w-screen h-screen md:w-[1250px] md:h-screen">
        <Spinner className="h-8 w-8 text-neutral-600" />
        <p className="text-neutral-600">Please Wait</p>
      </div>
    );
  }

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
