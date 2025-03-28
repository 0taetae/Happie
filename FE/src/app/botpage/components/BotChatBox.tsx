interface BotChatBoxProps {
  user: string;
  bot: string;
}

export default function BotChatBox({ user, bot }: BotChatBoxProps) {
  return (
    <div className="mt-6 w-full max-w-xl">
      {user && (
        <div className="p-4 bg-white border-l-4 border-blue-500 rounded shadow text-gray-800">
          <p>🙋 {user}</p>
        </div>
      )}
      {bot && (
        <div className="mt-4 p-4 bg-white border-l-4 border-green-500 rounded shadow text-gray-800">
          <p>🤖 하피의 응답: {bot}</p>
        </div>
      )}
    </div>
  );
}
