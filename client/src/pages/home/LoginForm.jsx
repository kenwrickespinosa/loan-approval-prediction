import { Input } from "@/components/ui/input";
import React, { useState } from "react";
import { CiLock } from "react-icons/ci";
import { CiUser } from "react-icons/ci";

function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleChangeEmail = (e) => {
    setEmail(e.target.value);
  }

  const handleChangePassword = (e) => {
    setPassword(e.target.value);
  }

  return (
    <div className="flex flex-col items-center gap-5 justify-center h-screen">
      <span className="font-semibold text-2xl text-neutral-600">Welcome</span>
      <form className="grid grid-rows-3 w-2xs gap-6">
        <div>
          <span className="flex items-center gap-1 text-neutral-500">
            <CiUser />
            Email
          </span>
          <Input value={email} onChange={handleChangeEmail} className="text-neutral-700" />
        </div>
        <div>
          <span className="flex items-center gap-1 text-neutral-500">
            <CiLock />
            Password
          </span>
          <Input value={password} onChange={handleChangePassword} className="text-neutral-700" />
        </div>
        <div
          className="text-center border border-blue-200 h-max rounded-md p-1 bg-blue-200
        text-blue-400 font-bold text-lg"
        >
          <button>Log In</button>
        </div>
      </form>
    </div>
  );
}

export default LoginForm;
