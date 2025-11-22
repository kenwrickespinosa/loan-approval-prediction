import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import LoginForm from "./pages/home";
import Dashboard from "./pages/dashboard";
import MainLayout from "./layouts/MainLayout";
import User from "./pages/user";
import Evaluate from "./pages/evaluate";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* All routes that share MainLayout */}
        <Route element={<MainLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/users" element={<User />} />
          <Route path="/evaluate" element={<Evaluate />} />
        </Route>

        {/* Login page without layout */}
        <Route path="/" element={<LoginForm />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
