import React from "react";

function UserTable({ clients }) {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-left">
          <th className="px-11 py-4">ID</th>
          <th className="px-11 py-4">Firstname</th>
          <th className="px-11 py-4">Lastname</th>
          <th className="px-11 py-4">Gender</th>
          <th className="px-11 py-4">Birthdate</th>
          <th className="px-11 py-4">Address</th>
          <th className="px-11 py-4">Contact Number</th>
        </tr>
      </thead>

      <tbody>
        {clients.map((client) => (
          <tr key={client.id} className="border-t">
            <td className="px-11 py-4">{client.id}</td>
            <td className="px-11 py-4">{client.firstname}</td>
            <td className="px-11 py-4">{client.lastname}</td>
            <td className="px-11 py-4">{client.gender}</td>
            <td className="px-11 py-4">{client.birthdate}</td>
            <td className="px-11 py-4">{client.address}</td>
            <td className="px-11 py-4">{client.contact_number}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default UserTable;
