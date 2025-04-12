import React, { useRef, useEffect, useState  } from 'react';
import '../assets/css/App.css';
import '../assets/css/MainPage.css';
import rainVideo from '../assets/videos/rain4.mp4';
import gpsImage from '../assets/img/gps_img.jpg';
import contact from '../assets/img/contactUs.jpg';
import elderly from '../assets/img/elderly3.jpg';
import vid2 from '../assets/videos/vid2.mp4';
import vid3 from '../assets/videos/vid3.mp4';

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
    phone: "",
    guardian: false,
    callRequest: false,
  });

  // 입력 값 변경 핸들러
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;

    setFormData((prevData) => ({
      ...prevData,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  // 폼 제출 핸들러
  const handleSubmit = (e) => {
    e.preventDefault(); // 기본 폼 제출 동작 방지
    console.log("Submitted Data:", formData); // 콘솔에 데이터 출력
    console.log("Submitted Data (JSON):", JSON.stringify(formData, null, 2)); // JSON 형태로 콘솔 출력
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
              <p>Protecting lives, securing vulnerable communities.</p>
              <p>We are dedicated to providing timely, accurate disaster alerts.</p>
            </div>
          </div>
        </div>
      </section>

      <section ref={serviceSectionRef} className="info-section">
        <div className="info-container">
          <div className="info-card">
            <div className="info-image">
              <img src={elderly} alt="Location Tracking" />
            </div>
            <div className="info-text">
              <h3>Emergency Alerts for the Vulnerable</h3>
              <p>Get real-time disaster alerts and phone assistance, tailored to the needs of vulnerable individuals, to stay safe during emergencies.</p>
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
              <h3>AI Report</h3>
              <p>
              Report emergencies on your behalf based on your personal situation.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">❄️</div>
              <h3>Shelter Map</h3>
              <p>
              Provides shelter information and guidance maps tailored to each type of natural disaster.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section ref={contactSectionRef} className="contact-section">
        <div className="contact-container">

          {/* Left Input Form */}
          <div className="input-section">
            <div className="contact-header">
              <h1>Contact Us</h1>
              <a href="mailto:your-email@example.com">Contact for bulk user support</a>
            </div>
            <form className="contact-form" onSubmit={handleSubmit}>

              {/* Who you are (Dropdown) */}
              <div className="form-group">
                <label htmlFor="personType">Who are you?</label>
                <select
                  id="personType"
                  name="personType"
                  value={formData.personType}
                  onChange={handleChange}
                  required
                >
                  <option value="" disabled selected>Select your situation</option>
                  <option value="elderly">Elderly / Living Alone / Pregnant / Guardian of Infant / Mobility Issues</option>
                  <option value="visual">Visually Impaired</option>
                  <option value="hearing">Hearing Impaired</option>
                  <option value="foreigner">Foreigner / Refugee</option>
                  <option value="other">Other</option>
                </select>
              </div>

              {/* Address */}
              <div className="form-group">
                <label htmlFor="address">Address</label>
                <input
                  type="text"
                  id="address"
                  name="address"
                  placeholder="Enter your full residential address"
                  value={formData.address}
                  onChange={handleChange}
                  required
                />
              </div>

              {/* International Phone Number */}
              <div className="form-group">
                <label htmlFor="phone">Phone Number (International)</label>
                <input
                  type="tel"
                  id="phone"
                  name="phone"
                  placeholder="e.g., +82 10-1234-5678"
                  value={formData.phone}
                  onChange={handleChange}
                  required
                />
              </div>

              {/* Guardian Connection */}
              <div className="form-group">
                <label>
                  <input
                    type="checkbox"
                    name="hasGuardian"
                    checked={formData.hasGuardian}
                    onChange={(e) => setFormData({ ...formData, hasGuardian: e.target.checked })}
                  />
                  &nbsp;I would like to connect a guardian
                </label>
                {formData.hasGuardian && (
                  <input
                    type="tel"
                    name="guardianPhone"
                    placeholder="Guardian's phone number (e.g., +82 10-0000-0000)"
                    value={formData.guardianPhone}
                    onChange={handleChange}
                    className="guardian-phone"
                  />
                )}
              </div>
              
              {/* Request Phone Guidance */}
              <div className="form-group">
                <label>
                  <input
                    type="checkbox"
                    name="needCall"
                    checked={formData.needCall}
                    onChange={(e) => setFormData({ ...formData, needCall: e.target.checked })}
                  />
                  &nbsp;I would like a guidance call from a support agent
                </label>
              </div>
              
              {/* Submit Button */}
              <button type="submit">Submit &gt;&gt;</button>
            </form>
          </div>
              
          {/* Right Image Section - Unchanged */}
          <div className="image-section">
            <img src={contact} alt="Contact Us" />
          </div>
              
        </div>
      </section>

      <footer className="footer-section">
        <div className="footer-container">
          {/* 왼쪽 영역 */}
          <div className="footer-left">
            <h3>Code Wave</h3>
          </div>

          {/* 오른쪽 영역 */}
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
