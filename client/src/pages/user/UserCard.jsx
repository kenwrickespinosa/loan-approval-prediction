import { Card, CardContent } from "@/components/ui/card";
import React from "react";
import { CiCalendar } from "react-icons/ci";
import { CiLocationOn } from "react-icons/ci";
import { CiPhone } from "react-icons/ci";
import { CiUser } from "react-icons/ci";

function UserCard({ client }) {
  const sliceAddress = (client) => {
    const newAddress = client.address.slice(0, 13)+"...";
    return newAddress;
  };

  return (
    <Card>
      <CardContent className="flex flex-col gap-6 p-6">
        <div className="flex flex-col">
          <span className="text-neutral-400 text-xs">FULL NAME</span>
          <span className="text-xl font-semibold">
            {client.firstname} {client.lastname}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-6">
          <div className="flex items-center gap-2">
            <CiUser className="text-blue-400 font-bold size-6" />
            <div className="flex flex-col">
              <span className="text-neutral-400 text-xs">GENDER</span>
              <span className="text-sm font-medium">{client.gender}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <CiCalendar className="text-blue-400 font-bold size-6" />
            <div className="flex flex-col">
              <span className="text-neutral-400 text-xs">BIRTHDATE</span>
              <span className="text-sm font-medium">{client.birthdate}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <CiLocationOn className="text-blue-400 font-bold size-6" />
            <div className="flex flex-col">
              <span className="text-neutral-400 text-xs">ADDRESS</span>
              <span className="text-sm font-medium">{sliceAddress(client)}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <CiPhone className="text-blue-400 font-bold size-6" />
            <div className="flex flex-col">
              <span className="text-neutral-400 text-xs">CONTACT NO.</span>
              <span className="text-sm font-medium">
                {client.contact_number}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default UserCard;
