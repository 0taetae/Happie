import "../globals.css";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";

export default function WebPageLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="flex flex-col h-screen">
        {/* 헤더 (상단 고정) */}
        <Header />

        {/* 메인 컨텐츠 영역 (사이드바 + 페이지 내용) */}
        <div className="flex flex-grow">
          {/* 왼쪽 사이드바 */}
          <Sidebar />

          {/* 오른쪽 컨텐츠 영역 */}
          <main className="flex-grow p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
