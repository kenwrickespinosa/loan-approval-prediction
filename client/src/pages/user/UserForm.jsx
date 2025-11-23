import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import React from "react";

function UserForm() {
  return (
    <div>
      <form className="flex flex-col gap-6">
        <div className="flex gap-6">
          <span>
            Firstname
            <Input type="text" />
          </span>
          <span>
            Lastname
            <Input type="text" />
          </span>
        </div>
        <div className="flex flex-col gap-6">
          <span>
            Address
            <Input type="text" />
          </span>
          <span>
            Contact Number
            <Input type="number" />
          </span>
        </div>
        <Button type="submit" className="cursor-pointer">
          Create User
        </Button>
      </form>
    </div>
  );
}

export default UserForm;
