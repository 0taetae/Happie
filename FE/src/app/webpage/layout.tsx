import "../globals.css";
import Sidebar from "./components/Sidebar";
import Map from "./components/Map";
import OrderButton from "./components/OrderButton";
import DrugHistoryButton from "./components/DrugHistoryButton";
import Warning from "./components/Warning";

export default function WebPageLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col h-screen">
      {/* 헤더 (상단 고정) */}
      <header className="bg-blue-200 flex justify-between p-4 text-lg font-bold shadow-md">
        <div className="text-white">🏥 하피 (happie)</div>
        <div className="text-white">한살차이</div>
      </header>

      {/* 메인 컨텐츠 영역 (사이드바 + 본문) */}
      <div className="flex flex-grow">
        {/* 왼쪽 사이드바 */}
        <Sidebar />

        {/* 오른쪽 컨텐츠 영역 */}
        <main className="flex flex-col flex-grow p-4">
          {/* 현재 페이지의 본문 */}
          <div>{children}</div>

          {/* 지도 및 ORDER 버튼 */}
          <div className="mt-4 flex flex-col items-center space-y-4">
            <Map />

            <div className="self-end">
              <div className="flex flex-row gap-3">
                <Warning />
                <DrugHistoryButton />
                <OrderButton />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
