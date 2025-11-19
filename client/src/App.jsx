import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import LoginForm from "./pages/home/LoginForm";
import Dashboard from "./pages/dashboard/Dashboard";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LoginForm />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
