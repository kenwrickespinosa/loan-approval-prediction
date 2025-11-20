import { Input } from "@/components/ui/input";
import React, { useEffect, useState } from "react";
import { CiLock } from "react-icons/ci";
import { CiUser } from "react-icons/ci";
import { useNavigate } from "react-router-dom";

function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleChangeEmail = (e) => {
    setEmail(e.target.value);
  };

  const handleChangePassword = (e) => {
    setPassword(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://127.0.0.1:8000/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Failed to login");
      }

      localStorage.setItem("token", data.token);
      navigate("/dashboard");
      console.log("Login successfully");
    } catch (error) {
      console.log("Error:", error);
    }
  };

  return (
    <div className="flex flex-col items-center gap-5 justify-center h-screen">
      <span className="font-semibold text-2xl text-neutral-600">Welcome</span>
      <form onSubmit={handleSubmit} className="grid grid-rows-3 w-2xs gap-6">
        <div>
          <span className="flex items-center gap-1 text-neutral-500">
            <CiUser />
            Email
          </span>
          <Input
            type="email"
            value={email}
            onChange={handleChangeEmail}
            className="text-neutral-700"
          />
        </div>
        <div>
          <span className="flex items-center gap-1 text-neutral-500">
            <CiLock />
            Password
          </span>
          <Input
            type="password"
            value={password}
            onChange={handleChangePassword}
            className="text-neutral-700"
          />
        </div>
        <div
          className="text-center border border-blue-200 h-max rounded-md p-1 bg-blue-200
        text-blue-400 font-bold text-lg"
        >
          <button type="submit">Log In</button>
        </div>
      </form>
    </div>
  );
}

export default LoginForm;
