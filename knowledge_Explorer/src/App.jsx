"use client";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import QueryMode from "./Components/QueryMode";
import InfoMode from "./Components/InfoMode";
import IntroPage from "./Components/IntroPage";
function App() {
  return (
    <>
      <Router>
      <Routes>
        <Route path="/" element={<IntroPage />} />
        <Route path="/query" element={<QueryMode />} />
        <Route path="/data" element={<InfoMode />} />
      </Routes>
    </Router>
    </>
  );
}

export default App;
