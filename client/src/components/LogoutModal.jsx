import React, { useState } from "react";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { useNavigate } from "react-router-dom";

function LogoutModal() {
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem("token");

      if (!token) {
        navigate("/");
        return;
      }

      const res = await fetch("http://127.0.0.1:8000/api/logout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || "Unable to Logout");
      }

      localStorage.removeItem("token");
      navigate("/");
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button className="bg-inherit text-black font-normal px-2 hover:cursor-pointer hover:bg-inherit">
          Logout
        </Button>
      </DialogTrigger>
      <DialogContent className="w-[300px] md:w-full">
        <DialogHeader>
          <DialogTitle>Are you sure you want to logout?</DialogTitle>
          <DialogDescription>
            Clicking yes will logout your account.
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-4 mt-6 justify-end">
          <div>
            <Button
              onClick={handleLogout}
              className="px-6 hover:cursor-pointer"
            >
              Yes
            </Button>
          </div>
          <DialogClose asChild>
            <Button variant="outline" className="hover:cursor-pointer">
              Cancel
            </Button>
          </DialogClose>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default LogoutModal;
