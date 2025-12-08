import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import React, { useEffect, useState } from "react";
import ClientSearchbar from "./ClientSearchbar";
import { Button } from "@/components/ui/button";

function PredictionForm() {
  const [gender, setGender] = useState("");
  const [married, setMarried] = useState("");
  const [dependents, setDependents] = useState("");
  const [education, setEducation] = useState("");
  const [selfEmployed, setSelfEmployed] = useState("");
  const [applicantIncome, setApplicantIncome] = useState("");
  const [coapplicantIncome, setCoapplicantIncome] = useState("");
  const [loanAmount, setLoanAmount] = useState("");
  const [loanAmountTerm, setLoanAmountTerm] = useState("");
  const [creditHistory, setCreditHistory] = useState("");
  const [propertyArea, setPropertyArea] = useState("");

  const [clientsList, setClientsList] = useState([]);
  const [selectedClient, setSelectedClient] = useState("");

  useEffect(() => {
    const fetchClients = async () => {
      const token = localStorage.getItem("token");

      try {
        const response = await fetch("http://127.0.0.1:8000/api/clients", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.message || "Failed to fetch clients");
        }

        setClientsList(data);
      } catch (error) {
        console.error("Error:", error);
      }
    };

    fetchClients();
  }, []);

  return (
    <div className="flex flex-col gap-6 border rounded-md p-4">
      <span className="flex justify-center text-neutral-500 font-bold">
        Evaluation Form for Loan Prediction
      </span>
      <form className="flex flex-col gap-4">
        <div>
          <ClientSearchbar
            clientsList={clientsList}
            onSelect={setSelectedClient}
          />
        </div>
        <div className="flex justify-between">
          <Select onValueChange={setGender}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Select Gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Gender</SelectLabel>
                <SelectItem value="Male">Male</SelectItem>
                <SelectItem value="Female">Female</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          <Select onValueChange={setMarried}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Marital Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Is client married?</SelectLabel>
                <SelectItem value="Yes">Married</SelectItem>
                <SelectItem value="No">Not Married</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="flex justify-between">
          <Select onValueChange={setEducation}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Educ. Level" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Education Level</SelectLabel>
                <SelectItem value="Graduate">Graduate</SelectItem>
                <SelectItem value="Not Graduate">Not Graduate</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          <Select onValueChange={setSelfEmployed}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Self Employed" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Is client self-employed?</SelectLabel>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="flex justify-between">
          <Select onValueChange={setDependents}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Dependents" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>No. of persons depending on client</SelectLabel>
                <SelectItem value="0">0</SelectItem>
                <SelectItem value="1">1</SelectItem>
                <SelectItem value="2">2</SelectItem>
                <SelectItem value="3">More than 3</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          <Select onValueChange={setCreditHistory}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Credit History" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Client meet guidelines?</SelectLabel>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Input
            value={applicantIncome}
            placeholder="Enter applicant income"
            onChange={(e) => setApplicantIncome(e.target.value)}
          />
        </div>
        <div>
          <Input
            value={coapplicantIncome}
            placeholder="Enter coapplicant Income"
            onChange={(e) => setCoapplicantIncome(e.target.value)}
          />
        </div>
        <div>
          <Input
            value={loanAmount}
            placeholder="Enter loan amount"
            onChange={(e) => setLoanAmount(e.target.value)}
          />
        </div>
        <div className="flex justify-between">
          <Select onValueChange={setPropertyArea}>
            <SelectTrigger className="w-[135px]">
              <SelectValue placeholder="Property Area" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Where applicant lives?</SelectLabel>
                <SelectItem value="Urban">Urban</SelectItem>
                <SelectItem value="Semi-Urban">Semi-Urban</SelectItem>
                <SelectItem value="Rural">Rural</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          <Input
            value={loanAmountTerm}
            placeholder="Term in months"
            onChange={(e) => setLoanAmountTerm(e.target.value)}
            className="w-[135px]"
          />
        </div>
        <Button>Evaluate</Button>
      </form>
    </div>
  );
}

export default PredictionForm;
