import React from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "./ui/sidebar";
import { Link, useLocation } from "react-router-dom";
import LogoutModal from "./LogoutModal";

const items = [
  {
    title: "Dashboard",
    url: "/dashboard",
  },
  {
    title: "Users - Applicants",
    url: "/users",
  },
  {
    title: "Evaluate",
    url: "/evaluate",
  },
];

function MainSidebar() {
  const location = useLocation();

  return (
    <Sidebar className="">
      <SidebarContent className="w-48 md:w-60">
        <SidebarGroup>
          <SidebarGroupLabel>Welcome</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item, index) => (
                <SidebarMenuItem key={index}>
                  <SidebarMenuButton isActive={location.pathname === item.url}>
                    <Link to={item.url} className="w-full">
                      {item.title}
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
            <SidebarMenu className="mb-4">
              <SidebarMenuItem>
                <LogoutModal />
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}

export default MainSidebar;
