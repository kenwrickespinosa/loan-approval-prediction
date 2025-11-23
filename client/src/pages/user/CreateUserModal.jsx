import React from "react";
import UserForm from "./UserForm";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

function CreateUserModal() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button className="cursor-pointer">Create User</Button>
      </DialogTrigger>
      <DialogContent className="md:w-max">
        <DialogHeader>
          <DialogTitle>Create User</DialogTitle>
          <DialogDescription />
        </DialogHeader>
        <UserForm />
      </DialogContent>
    </Dialog>
  );
}

export default CreateUserModal;
