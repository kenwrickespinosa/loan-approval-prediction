import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import React, { useState } from "react";

function ClientSearchbar({ clientsList, onSelect }) {
  const [query, setQuery] = useState("");
  const filteredClient = clientsList.filter((client) => {
    const q = query.toLowerCase();
    return (
      client.id.toString().includes(q) ||
      client.firstname.toLowerCase().includes(q) ||
      client.lastname.toLowerCase().includes(q)
    );
  });

  return (
    <Popover>
      <PopoverTrigger className="w-full">
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type/click here to search applicant"
        />
      </PopoverTrigger>
      <PopoverContent>
        <Command>
          <CommandInput placeholder="Search by ID or name" />
          <CommandList>
            {filteredClient.length === 0 && (
              <CommandEmpty>No Client Found.</CommandEmpty>
            )}
            {filteredClient.map((client) => (
              <CommandItem
                key={client.id}
                onSelect={() => {
                  onSelect(client.id);
                  setQuery(
                    `${client.firstname} ${client.lastname}`
                  );
                }}
              >
                {client.firstname} {client.lastname}, ID: {client.id}
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

export default ClientSearchbar;
