import React from "react";
import { useNavigate } from "react-router-dom";
const IntroPage = () => {
  const navigate = useNavigate();
  return (
    <>
      <div className="flex w-screen h-screen bg-white">
      <div className="w-full lg:w-1/2 flex items-center justify-center h-full">
        <div className="h-full w-full bg-gray-900 bg-[length:200%_200%] flex items-center justify-center">
          <div className="text-center p-8 max-w-lg w-full">
            <h1 className="text-5xl font-bold text-white mb-6 drop-shadow-lg">
              Meet <span className="font-extrabold text-amber-400 ">KHANIJ</span>
            </h1>
            <h2 className="text-white mb-4">
              <span className="font-bold">Khanij</span> is an AI
              assistant that can help you with your daily tasks.
            </h2>
            <p className="text-gray-200 text-lg mb-6">
              Choose one of the following modes to get started.
            </p>
            <div className="flex flex-col space-y-6">
              <button
                onClick={() => navigate("/query")}
                className="py-3 px-6 bg-white text-green-600 font-bold rounded-full shadow-lg hover:bg-green-100 transition duration-300"
              >
                Legal Explorer
              </button>
              <button
                onClick={() => navigate("/data")}
                className="py-3 px-6 bg-white text-blue-600 font-bold rounded-full shadow-lg hover:bg-blue-100 transition duration-300"
              >
                Mineral Explorer
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className="hidden lg:flex h-full w-1/2 justify-center items-center bg-gray-900">
        <img
          className="h-[500px]"
          src="https://i.ibb.co/vvv6wyP/frontpage.png"
          alt="riuy"
        />
      </div>
    </div>
    </>
  );
};

export default IntroPage;

//Query
// https://i.ibb.co/MBzCqQJ/query-Page.png