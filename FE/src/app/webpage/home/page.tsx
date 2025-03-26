export default function HomePage() {
  return (
    <div className="flex flex-col items-center">
      {/* 실시간 지도 */}
      <div className="w-full max-w-4xl p-4 bg-gray-200 rounded-lg shadow-md">
        <h2 className="text-lg font-bold mb-2">📍 실시간 로봇 위치</h2>
        <div className="bg-gray-300 h-80 flex items-center justify-center rounded-lg">
          <p className="text-gray-600">[실시간 지도 영역]</p>
        </div>
      </div>

      {/* ORDER 버튼 */}
      <button className="mt-4 px-6 py-2 bg-green-500 text-white font-bold rounded-lg shadow-lg hover:bg-green-700">
        ORDER
      </button>
    </div>
  );
}
