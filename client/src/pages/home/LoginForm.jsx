import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import React, { useEffect, useState } from "react";
import { CiLock } from "react-icons/ci";
import { CiUser } from "react-icons/ci";
import { useNavigate } from "react-router-dom";

function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState({});
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleChangeEmail = (e) => {
    setEmail(e.target.value);
    // setErrors((prev) => ({ ...prev, email: null }));
  };

  const handleChangePassword = (e) => {
    setPassword(e.target.value);
    // setErrors((prev) => ({ ...prev, password: null }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors({});

    try {
      setIsLoading(true);

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
        if (response.status === 422) {
          setErrors(data.errors || {});
          return;
        }
        throw new Error(data.message || "Failed to login");
      }

      localStorage.setItem("token", data.token);
      navigate("/dashboard");
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
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
          {errors.email && (
            <p className="text-red-500 mt-1 text-sm">{errors.email[0]}</p>
          )}
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
          {errors.password && (
            <p className="text-red-500 mt-1 text-sm">{errors.password[0]}</p>
          )}
        </div>
        {isLoading ? (
          <span className="flex justify-center items-center h-10">
            <Spinner className="size-6" />
          </span>
        ) : (
          <button
            type="submit"
            disabled={isLoading}
            className="text-center border border-blue-200 h-max rounded-md p-1 bg-blue-200
        text-blue-400 font-bold text-lg cursor-pointer"
          >
            Log In
          </button>
        )}
      </form>
    </div>
  );
}

export default LoginForm;
