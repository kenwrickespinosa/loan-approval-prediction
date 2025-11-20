import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import MainSidebar from "../components/MainSidebar";
import { Outlet } from "react-router-dom";

export default function MainLayout() {
  return (
    <SidebarProvider>
      <div className="md:grid md:grid-cols-[250px_1fr] h-screen">
        <aside>
          <SidebarTrigger />
          <MainSidebar />
        </aside>

        <main>
          <Outlet />
        </main>
      </div>
    </SidebarProvider>
  );
}
