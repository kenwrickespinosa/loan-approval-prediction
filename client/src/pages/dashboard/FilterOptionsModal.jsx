import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import React, { useEffect, useState } from "react";
import FilterOptions from "./FilterOptions";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { IoFilterSharp } from "react-icons/io5";

function FilterOptionsModal({ filters, setFilters }) {
  const [isOpen, setIsOpen] = useState(false);
  const [tempFilters, setTempFilters] = useState(filters);

  useEffect(() => {
    setTempFilters(filters);
  }, [filters]);

  const handleApply = () => {
    setFilters(tempFilters);
    setIsOpen(false);
  };

  return (
    <div className="text-right ml-6 md:ml-0 md:w-[1250px] pr-6">
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogTrigger asChild>
          <Button className="cursor-pointer mt-4 bg-white border-2 hover:bg-blue-200 hover:border-blue-200">
            <IoFilterSharp className="text-black font-bold size-6" />
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Apply Filter</DialogTitle>
            <VisuallyHidden>
              <DialogDescription />
            </VisuallyHidden>
          </DialogHeader>
          <div>
            <FilterOptions filters={tempFilters} onChange={setTempFilters} />
          </div>
          <DialogFooter>
            <Button onClick={handleApply}>Apply</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default FilterOptionsModal;
