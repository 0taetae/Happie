export default async function HomePage() {
  // 1. 지연 함수 만들기
  const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  // 2. 강제로 2초 기다리기
  await delay(200);

  return (
    <div className="flex">
    </div>
  );
}