import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const InfoMode = () => {
  const navigate = useNavigate();
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState("");
  const [showResponse, setShowResponse] = useState(false);

  const handleQuery = () => {
    setResponse("Work in Progress");
    setShowResponse(true);
  };

  return (
    <>
      <div className="flex w-screen h-screen bg-white">
        <div className="w-full lg:w-1/2 flex items-center justify-center h-full">
          <div className="h-full w-full bg-[#294457] bg-[length:200%_200%] flex items-center flex-col justify-center">
            <h1 className="text-5xl font-bold mb-10 text-[#FFE177]">Data Mode</h1>
            <input
              type="text"
              placeholder="Enter your query"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              className="p-2 mb-4 w-full max-w-md border-2 border-blue-400 rounded-md focus:outline-none focus:border-blue-600"
            />
            <button
              onClick={handleQuery}
              className="py-2 px-4 mb-6 bg-[#FFE177] text-white font-semibold rounded-full shadow-lg hover:bg-yellow-600 transition duration-300"
            >
              Submit
            </button>
            {showResponse && (
              <div className="w-full max-w-md space-y-4">
                <div className="p-4 bg-white rounded-lg shadow">
                  <h2 className="text-xl font-semibold mb-2">Response</h2>
                  <img className="h-[10rem] ml-28" src="https://i.ibb.co/25n9W8B/work-in-progress.png" alt="" />
                </div>
              </div>
            )}
            <button
              onClick={() => navigate("/")}
              className="mt-6 py-2 px-4 bg-gray-500 text-white font-semibold rounded-full shadow-lg hover:bg-gray-600 transition duration-300"
            >
              Change Mode
            </button>
          </div>
        </div>
        <div className="hidden lg:flex h-full w-1/2 justify-center items-center bg-[#294457]">
          <img
            className="h-[500px]"
            src="https://i.ibb.co/7C4RS5c/mineral-Exploration.png"
            alt="riuy"
          />
        </div>
      </div>
    </>
  );
};

export default InfoMode;
