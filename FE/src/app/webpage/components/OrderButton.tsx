"use client"; // 👈 클라이언트 컴포넌트로 지정

export default function OrderButton({ onClick }: { onClick?: () => void }) {
  return (
    <button
      className="mt-4 px-6 py-2 bg-green-500 text-white font-bold rounded-lg shadow-lg hover:bg-green-700 transition"
      onClick={onClick} // ✅ 이제 정상 작동함!
    >
      ORDER
    </button>
  );
}
