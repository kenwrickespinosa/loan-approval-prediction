import React from "react";
import UserCard from "./UserCard";
import UserTable from "./UserTable";

function UsersList({ clients }) {
  return (
    <div>
      {/** Mobile View */}
      <div className="flex justify-center px-4 py-4">
        <div className="flex flex-col gap-4 md:hidden w-full max-w-md">
          {clients.map((client) => (
            <UserCard key={client.id} client={client} />
          ))}
        </div>
      </div>

      {/** Desktop view */}
      <div className="hidden md:block">
        <UserTable clients={clients} />
      </div>
    </div>
  );
}

export default UsersList;
