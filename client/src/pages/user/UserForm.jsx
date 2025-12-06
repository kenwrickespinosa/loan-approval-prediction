import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import React, { useState } from "react";
import { IoMdMale } from "react-icons/io";
import { IoMdFemale } from "react-icons/io";

const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const years = Array.from({ length: 97 }, (_, i) => 1930 + i).reverse();

const days = Array.from({ length: 31 }, (_, i) => i + 1);

function UserForm() {
  const [firstname, setFirstname] = useState("");
  const [lastname, setLastname] = useState("");
  const [gender, setGender] = useState("Male");
  // const [birthdate, setBirthdate] = useState(new Date(2025, 5, 12));
  const [month, setMonth] = useState("");
  const [day, setDay] = useState("");
  const [year, setYear] = useState("");
  const [address, setAddress] = useState("");
  const [contact_number, setContactNumber] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!month || !day || !year) {
      console.log("Please select a complete birthdate.");
      return;
    }

    if (!firstname || !lastname || !gender || !address || !contact_number) {
      console.log("Please fill in all requested inputs.");
      return;
    }

    const formattedMonth = month.padStart(2, "0");
    const formattedDay = day.toString().padStart(2, "0");
    const birthdate = `${year}-${formattedMonth}-${formattedDay}`;

    try {
      const token = localStorage.getItem("token");

      const response = await fetch("http://127.0.0.1:8000/api/clients", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          firstname: firstname.trim(),
          lastname: lastname.trim(),
          gender,
          birthdate,
          address: address.trim(),
          contact_number: contact_number.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Failed to create a client");
      }
      console.log(data);
      setFirstname("");
      setLastname("");
      setMonth("");
      setDay("");
      setYear("");
      setAddress("");
      setContactNumber("");
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <div className="flex gap-6">
          <span>
            Firstname
            <Input
              type="text"
              value={firstname}
              onChange={(e) => setFirstname(e.target.value)}
            />
          </span>
          <span>
            Lastname
            <Input
              type="text"
              value={lastname}
              onChange={(e) => setLastname(e.target.value)}
            />
          </span>
        </div>
        <div className="grid grid-cols-2 gap-6">
          <Button
            type="button"
            onClick={() => setGender("Male")}
            className={`${
              gender === "Male" ? "bg-blue-200" : "bg-white"
            } border text-black md:hover:bg-blue-200`}
          >
            <IoMdMale />
            Male
          </Button>
          <Button
            type="button"
            onClick={() => setGender("Female")}
            className={`${
              gender === "Female" ? "bg-red-200" : "bg-white"
            } border text-black md:hover:bg-red-200`}
          >
            <IoMdFemale />
            Female
          </Button>
        </div>
        <div className="flex flex-col gap-6">
          <span>
            Address
            <Input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
            />
          </span>
          <span>
            Contact Number
            <Input
              type="number"
              value={contact_number}
              onChange={(e) => setContactNumber(e.target.value)}
            />
          </span>
        </div>
        <div>
          <span>Birthdate</span>
          <div className="flex justify-between">
            <Select value={month} onValueChange={setMonth}>
              <SelectTrigger className="w-[100px] md:w-[125px]">
                <SelectValue placeholder="Month" />
              </SelectTrigger>
              <SelectContent className="max-h-60 overflow-y-auto">
                {months.map((m, i) => (
                  <SelectItem key={i} value={(i + 1).toString()}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={day} onValueChange={setDay}>
              <SelectTrigger className="w-[100px] md:w-[125px]">
                <SelectValue placeholder="Day" />
              </SelectTrigger>
              <SelectContent className="max-h-60 overflow-y-auto">
                {days.map((d) => (
                  <SelectItem key={d} value={d.toString()}>
                    {d}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={year} onValueChange={setYear}>
              <SelectTrigger className="w-[100px] md:w-[125px]">
                <SelectValue placeholder="Year" />
              </SelectTrigger>
              <SelectContent className="max-h-60 overflow-y-auto">
                {years.map((y) => (
                  <SelectItem key={y} value={y.toString()}>
                    {y}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <Button type="submit">Create</Button>
      </form>
    </div>
  );
}

export default UserForm;
