import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import React from "react";

function FilterOptions({ filters, onChange }) {
  return (
    <div>
      <p>Filter Options</p>
      <div>
        <p>Gender</p>
        <RadioGroup
          value={filters.gender}
          onValueChange={(val) =>
            onChange((prev) => ({ ...prev, gender: val }))
          }
        >
          <div>
            <RadioGroupItem value="all" id="g1" />
            <Label htmlFor="g1">All</Label>
          </div>
          <div>
            <RadioGroupItem value="Male" id="g2" />
            <Label htmlFor="g2">Male</Label>
          </div>
          <div>
            <RadioGroupItem value="Female" id="g3" />
            <Label htmlFor="g3">Female</Label>
          </div>
        </RadioGroup>
      </div>
      <div>
        <p>Marital Status</p>
        <RadioGroup
          value={filters.married}
          onValueChange={(val) =>
            onChange((prev) => ({ ...prev, married: val }))
          }
        >
          <div>
            <RadioGroupItem value="all" id="m1" />
            <Label htmlFor="m1">All</Label>
          </div>
          <div>
            <RadioGroupItem value="Yes" id="m2" />
            <Label htmlFor="m2">Married</Label>
          </div>
          <div>
            <RadioGroupItem value="No" id="m3" />
            <Label htmlFor="m3">Not Married</Label>
          </div>
        </RadioGroup>
      </div>
    </div>
  );
}

export default FilterOptions;
