import { Input } from "@/components/ui/input";
import React from "react";

function LoginForm() {
  return (
    <div className="flex flex-col items-center">
      <span>Log In</span>
      <div className="grid grid-rows-3 w-2xs">
        <div className="">
          <span>Email</span>
          <Input />
        </div>
        <div>
          <span>Password</span>
          <Input />
        </div>
        <div>
          <button>Log In</button>
        </div>
      </div>
    </div>
  );
}

export default LoginForm;
