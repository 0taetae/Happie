import "../app/globals.css";
import { ReactNode } from "react";

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="ko">
      <body className="bg-gray-100 text-gray-900">{children}</body>
    </html>
  );
}
