import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './assets/css/App.css';
import MainPage from './component/MainPage'; // MainPage import

function App() {
  // 스크롤 효과
  useEffect(() => {
    const handleScroll = () => {
      const appElement = document.querySelector('.app');
      if (appElement) {
        if (window.scrollY > 100) {
          appElement.classList.add('scrolled');
        } else {
          appElement.classList.remove('scrolled');
        }
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // React Element 반환
  return (
    <Router>
      <div className="app">
        <main className="content" style={{ minHeight: '940px' }}>
          <Routes>
            <Route path="/" element={<MainPage />} />
            {/* 앞으로 필요한 다른 페이지도 여기에 추가하면 됨 */}
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;