import React, { useRef, useEffect, useState  } from 'react';
import '../assets/css/App.css';
import '../assets/css/MainPage.css';
import rainVideo from '../assets/videos/rain4.mp4';
import gpsImage from '../assets/img/gps_img.jpg';
import contact from '../assets/img/contactUs.jpg';

function MainPage() {

  // 변수 설정
  const homeSectionRef = useRef(null);
  const aboutSectionRef = useRef(null);
  const serviceSectionRef = useRef(null);
  const contactSectionRef = useRef(null);
  const videoRef = useRef(null);

  // 스크롤 이동 함수
  const scrollToSection = (sectionRef) => {
    if (sectionRef.current) {
      sectionRef.current.scrollIntoView({
        behavior: "smooth", // 부드러운 스크롤 효과
        block: "start", // 섹션의 상단을 화면에 맞춤
      });
    }
  };  

  // 백그라운드 비디오 재생
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const loopSection = () => {
      if (video.currentTime >= 30) {
        video.currentTime = 0;
        video.play();
      }
    };
    video.addEventListener('timeupdate', loopSection);

    return () => {
      video.removeEventListener('timeupdate', loopSection);
    };
  }, []);

  // 폼 데이터를 관리하는 상태
  const [formData, setFormData] = useState({
    personType: "",
    address: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // 입력 값 변경 핸들러
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  // 폼 제출 핸들러
  const handleSubmit = (e) => {
    e.preventDefault(); // 기본 폼 제출 동작 방지
    console.log("Submitted Data:", formData); // 콘솔에 데이터 출력
  };

  return (
    <div>

      <div ref={homeSectionRef} className="hero-container">
        {/* 좌측 상단 로고 */}
        <div className="navbar-container">
          <div className="navbar-content">
            <div className="logo">DisasterAlert</div>
            <div className="nav-links">
              {/* <a href="/">Home</a>
              <a href="/">About</a>
              <a href="/">Service</a>
              <a href="/">Contact</a> */}
              <a onClick={() => scrollToSection(homeSectionRef)}>Home</a>
              <a onClick={() => scrollToSection(aboutSectionRef)}>About</a>
              <a onClick={() => scrollToSection(serviceSectionRef)}>Service</a>
              <a onClick={() => scrollToSection(contactSectionRef)}>Contact</a>
            </div>
          </div>
        </div>

        {/* 배경 빗방울 영상 */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="background-video"
        >
          <source src={rainVideo} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="video-overlay" /> {/* 어두운 레이어 */}

        <div className="hero-content">
          <h1>Stay Safe, Stay<br />Informed</h1>
          <p>Receive immediate alerts about natural disasters near you.<br />We help you stay safe and informed.</p>
          <button className="cta-button" onClick={() => scrollToSection(contactSectionRef)}>Contact Us</button>
        </div>
      </div>

      <section ref={aboutSectionRef} className="our-story-section">
        <div className="our-story-container">
          <div className="our-story-content">
            <div className="our-story-title">
              <h2>Our Vision</h2>
            </div>
            <div className="our-story-description">
              <p>Protecting lives, securing communities.</p>
              <p>We are dedicated to providing timely, accurate disaster alerts.</p>
            </div>
          </div>
        </div>
      </section>

      <section ref={serviceSectionRef} className="info-section">
        <div className="info-container">
          <div className="info-card">
            <div className="info-image">
              <img src={gpsImage} alt="Location Tracking" />
            </div>
            <div className="info-text">
              <h3>Precise Location Tracking</h3>
              <p>
                Real-time alerts and precise location tracking offer unparalleled safety during critical moments of natural disasters.
              </p>
              <button className="info-button" onClick={() => scrollToSection(contactSectionRef)}>Contact Us</button>
            </div>
          </div>
        </div>
      </section>

      <section className="features-section">
        <div className="features-inner">
          <h2 className="features-title">Key Features</h2>
          <div className="features-cards">
            <div className="feature-card">
              <div className="feature-icon">❄️</div>
              <h3>Rapid Alerts</h3>
              <p>
                Get instant notifications about impending disasters, ensuring you're always one step ahead.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">❄️</div>
              <h3>Live Tracking</h3>
              <p>
                Access expert advice on how to prepare for and respond to various natural disasters effectively.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">❄️</div>
              <h3>Safety Tips</h3>
              <p>
                Access expert advice on how to prepare for and respond to various natural disasters effectively.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section ref={contactSectionRef} className="contact-section">
        <div className="contact-container">
          {/* 왼쪽 입력 폼 */}
          <div className="input-section">
            <h1>Contact Us</h1>
            <form className="contact-form" onSubmit={handleSubmit}>
              {/* 전화번호 입력 */}
              <div className="form-group">
                <label htmlFor="phone">Phone Number</label>
                <div className="phone-input">
                  <select
                    id="country-code"
                    name="phone"
                    value={formData.phone}
                    onChange={handleChange}
                  >
                    <option value="+82">South Korea (+82)</option>
                    <option value="+1">United States (+1)</option>
                  </select>
                  <input
                    type="tel"
                    id="phone"
                    name="phone"
                    placeholder="Enter your phone number"
                    required
                    onChange={handleChange}
                  />
                </div>
              </div>

              {/* 이메일 입력 */}
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  placeholder="Enter your email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
              </div>

              {/* 장애 여부 선택 */}
              <div className="form-group">
                <label htmlFor="disability">Disability Status</label>
                <select
                  id="disability"
                  name="disability"
                  value={formData.disability}
                  onChange={handleChange}
                >
                  <option value="">Select an option</option>
                  <option value="visual">Visual Impairment</option>
                  <option value="hearing">Hearing Impairment</option>
                  <option value="mobility">Mobility Difficulty</option>
                </select>
              </div>

              {/* 집 주소 입력 */}
              <div className="form-group">
                <label htmlFor="address">Home Address</label>
                <textarea
                  id="address"
                  name="address"
                  rows="3"
                  placeholder="Enter your full address"
                  value={formData.address}
                  onChange={handleChange}
                  required
                ></textarea>
              </div>

              {/* 제출 버튼 */}
              <button type="submit">Submit &gt;&gt;</button>
            </form>
          </div>

          {/* 오른쪽 이미지 영역 */}
          <div className="image-section">
            <img src={contact} alt="Contact Us" />
          </div>
        </div>
      </section>

      <footer className="footer-section">
        <div className="footer-container">
          <div className="footer-left">
            <h3>Code Wave</h3>
          </div>

          <div className="footer-right">
            <ul>
              <li>
                <a href="https://github.com/your-github-profile" target="_blank" rel="noopener noreferrer">
                  GitHub
                </a>
              </li>
              <li>
                <a href="mailto:your-email@example.com">Email</a>
              </li>
            </ul>
          </div>
        </div>
      </footer>

    </div>
  );
}

export default MainPage;
