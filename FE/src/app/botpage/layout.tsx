export default function BotLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      {children}
    </div>
  );
}
